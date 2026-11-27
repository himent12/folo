//! `/api/health` and `/api/system`: health, system info, the throughput table and benchmark probes.

use axum::Json;
use axum::extract::State;
use folo_api::{
    API_VERSION, BenchmarkReport, BenchmarkRequest, Health, SystemInfo, SystemResponse, ThroughputTable,
};

use std::sync::Arc;

use tokio::sync::watch;

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
        crate::store::FEATURE_PREDICT_OPTIONS,
        crate::store::FEATURE_VAL_DIAGNOSTICS,
        crate::store::FEATURE_EXPORT_DYNAMIC_AXES,
        crate::store::FEATURE_TRAIN_AUTO_BATCH,
        crate::store::FEATURE_FOLO2_ARCH,
        crate::store::FEATURE_THRESHOLD_CALIBRATION,
        crate::store::FEATURE_CLOUD_SPOT_RESUME,
    ];
    let labeling = st.store.labeler().is_ok();
    let optional = [
        (st.web_wasm, folo_api::FEATURE_WEB_WASM),
        (labeling, folo_api::FEATURE_AGENT_LABELING),
        (labeling, folo_api::FEATURE_LABEL_IMAGE_FILTERS),
        (labeling, folo_api::FEATURE_LABEL_REVISIONS),
        (labeling, folo_api::FEATURE_LABEL_IMAGE_PATH),
        (st.store.benches().is_ok(), folo_api::FEATURE_VISION_BENCH),
        (st.store.benches().is_ok(), crate::store::FEATURE_BENCH_UPLOAD_SAFE),
        (st.store.benches().is_ok(), crate::store::FEATURE_BENCH_GENERATED_SOURCE),
    ];
    always
        .iter()
        .chain(st.store.features())
        .copied()
        .chain(optional.into_iter().filter(|(on, _)| *on).map(|(_, f)| f))
        .map(str::to_string)
        .collect()
}

#[derive(Debug, serde::Serialize)]
pub(super) struct SystemWithSimd {
    #[serde(flatten)]
    base: SystemResponse,
    cpu_simd: crate::system::CpuSimd,
}

pub(super) async fn system(State(st): State<AppState>) -> Result<Json<SystemWithSimd>> {
    let prepare = st.clone();
    st.system_ready.ready(move || prepare_system(prepare)).await?;
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
    let base = SystemResponse {
        system: SystemInfo { cpu, memory, gpus },
        data_dir,
        features,
        capabilities,
        gpu_pool,
    };
    Ok(Json(SystemWithSimd { base, cpu_simd: crate::system::cpu_simd() }))
}

/// The slow parts of the first `/api/system`, side by side, each a blocking task of its own: creating the GPU device
/// (`capabilities`), enumerating the adapters, and the CPU counters' set-up and first interval. Run by
/// [`Preparation`] ([`AppState`]'s `system_ready`). Ends only once all three have ended; a part that panicked fails
/// it.
async fn prepare_system(st: AppState) -> Result<()> {
    let (store, adapters_store, probe) = (st.store.clone(), st.store.clone(), st.probe.clone());
    let device = tokio::task::spawn_blocking(move || drop(store.capabilities()));
    let adapters = tokio::task::spawn_blocking(move || adapters_store.prepare_gpus());
    let counters = tokio::task::spawn_blocking(move || probe.warm_up());
    let (device, adapters, counters) = tokio::join!(device, adapters, counters);
    device.and(adapters).and(counters).map_err(|e| Error::internal(format!("task failed: {e}")))
}

/// `/api/system`'s one-time preparation, owned by the server rather than by the request that started it. The first
/// request starts it as a task of its own; every request until it ends waits for its one result, without holding a
/// blocking thread. A request that goes away meanwhile (its client disconnected) changes nothing: the task goes on,
/// and the next request waits for the same run. After a run that failed (and so ended, with all its jobs), the next
/// request starts another; after one that succeeded, requests wait for nothing.
///
/// Its state belongs to one tokio runtime's life, as `folo serve`'s router does. A router used on a new runtime
/// after the old one shut down mid-run starts a new run, while blocking jobs of the old one may still be running.
#[derive(Default)]
pub(crate) struct Preparation {
    state: std::sync::Mutex<Prepared>,
    /// Requests that took the running preparation's receiver (and wait on it next). For tests.
    #[cfg(test)]
    waiters: std::sync::atomic::AtomicUsize,
}

/// Where [`Preparation`] is.
#[derive(Default)]
enum Prepared {
    /// Not run yet, or the last run failed.
    #[default]
    Not,
    /// Running: its result arrives here.
    Running(watch::Receiver<Option<Result<()>>>),
    Done,
}

impl Preparation {
    /// Waits until the preparation succeeded, starting it with `start` when it isn't running and hasn't succeeded;
    /// the error of the run waited for, if it failed. A caller that stops waiting leaves the run going.
    pub(crate) async fn ready<F>(self: &Arc<Self>, start: impl FnOnce() -> F) -> Result<()>
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        let mut result = {
            let mut prepared = self.state.lock().unwrap_or_else(|e| e.into_inner());
            match &*prepared {
                Prepared::Done => return Ok(()),
                // A run whose task went without a result (the runtime shutting down) is no run.
                Prepared::Running(result) if result.has_changed().is_ok() => result.clone(),
                _ => {
                    let (tx, result) = watch::channel(None);
                    let run = start();
                    let this = self.clone();
                    tokio::spawn(async move {
                        let outcome = run.await;
                        *this.state.lock().unwrap_or_else(|e| e.into_inner()) =
                            if outcome.is_ok() { Prepared::Done } else { Prepared::Not };
                        tx.send_replace(Some(outcome));
                    });
                    *prepared = Prepared::Running(result.clone());
                    result
                }
            }
        };
        #[cfg(test)]
        self.waiters.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let outcome = result
            .wait_for(Option::is_some)
            .await
            .map_err(|_| Error::internal("the system preparation stopped without a result"))?;
        outcome.clone().unwrap_or(Ok(()))
    }
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    fn state() -> AppState {
        AppState {
            store: Arc::new(crate::MockStore::new()),
            probe: Arc::new(crate::system::SystemProbe::default()),
            system_ready: Arc::default(),
            shutdown: Arc::new(tokio::sync::watch::channel(false).0),
            web_wasm: false,
        }
    }

    /// Generous upper bound for coordination waits under heavy CI load.
    const WAIT_LIMIT: Duration = Duration::from_secs(60);

    /// Waits (polling, up to a generous limit for a loaded machine) until `st`'s preparations started `n`
    /// warm-ups of the CPU counters.
    async fn warm_ups_reach(st: &AppState, n: usize) {
        let t = std::time::Instant::now();
        while st.probe.warm_ups.load(Ordering::SeqCst) < n {
            assert!(t.elapsed() < WAIT_LIMIT, "no warm-up started");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// Waits until `n` `/api/system` calls have attached to the running preparation's watch channel.
    async fn waiters_reach(st: &AppState, n: usize) {
        let t = std::time::Instant::now();
        while st.system_ready.waiters.load(Ordering::SeqCst) < n {
            assert!(t.elapsed() < WAIT_LIMIT, "waiters did not reach {n}");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    /// A burst of first `/api/system` calls, all in flight at once, shares one preparation, and nothing waits on a
    /// blocking thread for another blocking job: with a single blocking thread every call still answers. Later
    /// calls prepare nothing.
    #[test]
    fn first_calls_share_one_preparation_and_hold_no_blocking_thread_while_waiting() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let st = state();
        rt.block_on(async {
            // The preparation can't finish while the test holds the gate: every call is started meanwhile.
            let gate = st.probe.gate.close();
            let burst: Vec<_> = (0..16).map(|_| tokio::spawn(system(State(st.clone())))).collect();
            warm_ups_reach(&st, 1).await;
            // Every call has entered the wait for the preparation's result.
            waiters_reach(&st, 16).await;
            assert!(
                burst.iter().all(|call| !call.is_finished()),
                "a call answered before the preparation ended"
            );
            drop(gate);
            let answers = tokio::time::timeout(WAIT_LIMIT, async {
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
                assert!(r.base.system.cpu.threads >= 1 && r.base.capabilities.is_some());
                assert!(!r.cpu_simd.int8_kernel.is_empty());
            }
            assert_eq!(st.probe.warm_ups.load(Ordering::SeqCst), 1);
            for _ in 0..3 {
                assert!(system(State(st.clone())).await.is_ok());
            }
            assert_eq!(st.probe.warm_ups.load(Ordering::SeqCst), 1, "prepared once");
        });
    }

    /// The first request goes away (its client disconnected) while the preparation's jobs run: the next request
    /// waits for those jobs instead of starting three more beside them.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancelled_first_call_leaves_its_preparation_to_the_next() {
        let st = state();
        let gate = st.probe.gate.close();
        let first = tokio::spawn(system(State(st.clone())));
        warm_ups_reach(&st, 1).await;
        waiters_reach(&st, 1).await;
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        let second = tokio::spawn(system(State(st.clone())));
        // Deterministically confirm `second` attached to the running preparation without starting another.
        waiters_reach(&st, 2).await;
        assert_eq!(st.probe.warm_ups.load(Ordering::SeqCst), 1, "a second preparation started");
        drop(gate);
        let Json(r) = tokio::time::timeout(WAIT_LIMIT, second)
            .await
            .expect("second call answered")
            .unwrap()
            .unwrap_or_else(|e| panic!("{e:?}"));
        assert!(r.base.system.cpu.threads >= 1 && r.base.capabilities.is_some());
        assert!(!r.cpu_simd.int8_kernel.is_empty());
        assert_eq!(st.probe.warm_ups.load(Ordering::SeqCst), 1, "prepared once");
    }

    /// A run that failed fails every request waiting for it, and the next request starts another; once one
    /// succeeded, nothing is started again.
    #[tokio::test]
    async fn a_failed_preparation_is_run_again_and_a_successful_one_is_not() {
        let prep = Arc::new(Preparation::default());
        let runs = Arc::new(AtomicUsize::new(0));
        let run = |ok: bool| {
            let runs = runs.clone();
            move || async move {
                runs.fetch_add(1, Ordering::SeqCst);
                tokio::task::yield_now().await;
                if ok { Ok(()) } else { Err(Error::internal("failed on purpose")) }
            }
        };
        let (a, b) = tokio::join!(prep.ready(run(false)), prep.ready(run(true)));
        assert!(a.is_err() && b.is_err(), "both waited for the one run, which failed");
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(prep.ready(run(true)).await.is_ok());
        assert_eq!(runs.load(Ordering::SeqCst), 2);
        assert!(prep.ready(run(false)).await.is_ok(), "done: not run again");
        assert_eq!(runs.load(Ordering::SeqCst), 2);
    }
}
