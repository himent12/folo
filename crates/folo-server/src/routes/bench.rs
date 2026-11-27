//! The `/api/bench` handlers (docs/API.md, *Vision benchmarks*): glue between HTTP and [`Benches`],
//! which every store that has benchmarks hands out ([`crate::Store::benches`]; the mock has none: `501`).

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use folo_api::{
    Bench, BenchCatalog, BenchDesign, BenchEntry, BenchEntryImagePage, BenchEntryList, BenchImagePage,
    BenchList, BenchLoop, BenchLoopList, BenchPredictions, BenchResult, CreateBench, DesignBench,
    StartBenchEntry, StartBenchLoop, UpdateBench,
};
use folo_bench::Condition;
use futures_util::{Stream, StreamExt, stream};
use serde::Deserialize;
use tokio::sync::broadcast;
use tokio::sync::broadcast::error::RecvError;

use crate::AppState;
use crate::bench::{Benches, CreateGeneratedBench};
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::routes::{DEFAULT_LIMIT, KEEP_ALIVE, MAX_LIMIT, MAX_UPLOAD_BYTES, blocking, serve_image};
use crate::store::SseFrame;

/// The `/bench` routes, merged into the `/api` router.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/bench/catalog", get(catalog))
        .route("/bench/benchmarks", get(list).post(create))
        .route("/bench/benchmarks/{id}", get(get_bench).put(update).delete(delete))
        .route("/bench/benchmarks/{id}/design", post(design))
        .route("/bench/benchmarks/{id}/images", get(images))
        .route("/bench/benchmarks/{id}/images/{image_id}", get(image))
        .route(
            "/bench/benchmarks/{id}/entries",
            get(entries).post(start_entry).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route("/bench/benchmarks/{id}/entries/{entry_id}", get(entry).delete(delete_entry))
        .route("/bench/benchmarks/{id}/entries/{entry_id}/result", get(result))
        .route("/bench/benchmarks/{id}/entries/{entry_id}/images", get(entry_images))
        .route("/bench/benchmarks/{id}/entries/{entry_id}/stop", post(stop_entry))
        .route(
            "/bench/benchmarks/{id}/entries/{entry_id}/predictions",
            post(add_predictions).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route("/bench/benchmarks/{id}/entries/{entry_id}/finish", post(finish_entry))
        .route("/bench/benchmarks/{id}/loops", get(loops).post(start_loop))
        .route("/bench/benchmarks/{id}/loops/{loop_id}", get(get_loop).delete(delete_loop))
        .route("/bench/benchmarks/{id}/loops/{loop_id}/stop", post(stop_loop))
        .route("/bench/benchmarks/{id}/events", get(events))
}

fn bad_query(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_query", message)
}

/// `condition` and `severity` of a query: both or neither (empty values count as absent).
fn variant(condition: Option<String>, severity: Option<String>) -> Result<Option<(Condition, u8)>> {
    let condition = condition.filter(|c| !c.trim().is_empty());
    let severity = severity.filter(|s| !s.trim().is_empty());
    match (condition, severity) {
        (None, None) => Ok(None),
        (Some(c), Some(s)) => {
            let cond = Condition::parse(c.trim()).ok_or_else(|| {
                bad_query(format!("unknown condition '{c}'"))
                    .hint("GET /api/bench/catalog lists the conditions")
            })?;
            let sev = s
                .trim()
                .parse::<u8>()
                .ok()
                .filter(|v| (1..=folo_api::BENCH_SEVERITIES).contains(v))
                .ok_or_else(|| {
                    bad_query(format!("severity must be in 1..={}, got '{s}'", folo_api::BENCH_SEVERITIES))
                })?;
            Ok(Some((cond, sev)))
        }
        _ => Err(bad_query("condition and severity go together: give both or neither")),
    }
}

async fn catalog(State(st): State<AppState>) -> Result<Json<BenchCatalog>> {
    Ok(Json(st.store.benches()?.catalog()))
}

async fn list(State(st): State<AppState>) -> Result<Json<BenchList>> {
    Ok(Json(BenchList { benchmarks: st.store.benches()?.list() }))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateGeneratedBody {
    name: String,
    #[serde(default)]
    description: String,
    source: GeneratedSourceBody,
    #[serde(default)]
    max_images: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    plan: Option<folo_api::BenchPlan>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GeneratedSourceBody {
    kind: String,
    #[serde(default)]
    images: Option<u32>,
    #[serde(default)]
    classes: Option<Vec<String>>,
    #[serde(default)]
    imgsz: Option<u32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    vary_aspect: bool,
}

fn parse_generated_bench(body: serde_json::Value) -> Result<CreateGeneratedBench> {
    let raw: CreateGeneratedBody = serde_json::from_value(body)
        .map_err(|e| Error::new(StatusCode::BAD_REQUEST, "invalid_body", e.to_string()))?;
    debug_assert_eq!(raw.source.kind, "generated");
    let images = match (raw.source.images, raw.max_images) {
        (Some(a), Some(b)) if a != b => {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                format!("source.images ({a}) and max_images ({b}) disagree: give one or the same count"),
            ));
        }
        (Some(n), _) | (_, Some(n)) => n,
        (None, None) => folo_bench::DEFAULT_IMAGES,
    };
    let seed = match (raw.source.seed, raw.seed) {
        (Some(a), Some(b)) if a != b => {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                format!("source.seed ({a}) and seed ({b}) disagree: give one or the same seed"),
            ));
        }
        (Some(s), _) | (_, Some(s)) => s,
        (None, None) => 0,
    };
    let classes = raw
        .source
        .classes
        .unwrap_or_else(|| folo_bench::DEFAULT_CLASSES.iter().map(|s| (*s).to_string()).collect());
    let imgsz = raw.source.imgsz.unwrap_or(folo_bench::DEFAULT_IMGSZ);
    Ok(CreateGeneratedBench {
        name: raw.name,
        description: raw.description,
        source: folo_bench::GeneratedSourceSpec {
            images,
            classes,
            imgsz,
            seed,
            vary_aspect: raw.source.vary_aspect,
        },
        plan: raw.plan,
    })
}

async fn create(State(st): State<AppState>, ApiJson(body): ApiJson<serde_json::Value>) -> Result<Response> {
    let is_generated = body.get("source").and_then(|s| s.get("kind")).and_then(serde_json::Value::as_str)
        == Some("generated");
    let bench: Bench = if is_generated {
        let req = parse_generated_bench(body)?;
        blocking(&st, move |s| s.benches()?.create_generated(req)).await?
    } else {
        let req: CreateBench = serde_json::from_value(body)
            .map_err(|e| Error::new(StatusCode::BAD_REQUEST, "invalid_body", e.to_string()))?;
        blocking(&st, move |s| s.benches()?.create(s, req)).await?
    };
    Ok((StatusCode::CREATED, Json(bench)).into_response())
}

async fn get_bench(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Json<Bench>> {
    Ok(Json(st.store.benches()?.get(&id)?))
}

async fn update(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<UpdateBench>,
) -> Result<Json<Bench>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.update(&id, req)).await?))
}

async fn delete(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<StatusCode> {
    blocking(&st, move |s| s.benches()?.delete(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn design(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<DesignBench>,
) -> Result<Json<BenchDesign>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.design(s, &id, req)).await?))
}

#[derive(Deserialize)]
struct ImagesParams {
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
    condition: Option<String>,
    severity: Option<String>,
}

async fn images(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<ImagesParams>,
) -> Result<Json<BenchImagePage>> {
    let v = variant(p.condition, p.severity)?;
    let limit = p.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
    Ok(Json(st.store.benches()?.images(&id, p.offset, limit, v)?))
}

#[derive(Deserialize)]
struct ImageParams {
    condition: Option<String>,
    severity: Option<String>,
    thumb: Option<u32>,
}

async fn image(
    State(st): State<AppState>,
    ApiPath((id, image_id)): ApiPath<(String, String)>,
    ApiQuery(p): ApiQuery<ImageParams>,
    headers: HeaderMap,
) -> Result<Response> {
    let v = variant(p.condition, p.severity)?;
    let thumb = p.thumb.map(|t| t.clamp(16, 2048));
    serve_image(&st, &headers, move |s| s.benches()?.image(&id, &image_id, v, thumb)).await
}

async fn entries(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Json<BenchEntryList>> {
    Ok(Json(BenchEntryList { entries: st.store.benches()?.entries(&id)? }))
}

#[derive(Deserialize)]
struct StartEntryParams {
    #[serde(default)]
    entry_id: Option<String>,
}

async fn start_entry(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<StartEntryParams>,
    ApiJson(req): ApiJson<StartBenchEntry>,
) -> Result<Response> {
    // An external entry is scored in the request: on the blocking pool, as everything else here.
    let entry: BenchEntry =
        blocking(&st, move |s| s.benches()?.start_entry_with_id(s, &id, req, p.entry_id)).await?;
    Ok((StatusCode::CREATED, Json(entry)).into_response())
}

async fn entry(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
) -> Result<Json<BenchEntry>> {
    Ok(Json(st.store.benches()?.entry(&id, &entry_id)?))
}

async fn delete_entry(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.benches()?.delete_entry(&id, &entry_id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn result(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
) -> Result<Json<BenchResult>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.result(&id, &entry_id)).await?))
}

#[derive(Deserialize)]
struct EntryImagesParams {
    sort: Option<String>,
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
}

async fn entry_images(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
    ApiQuery(p): ApiQuery<EntryImagesParams>,
) -> Result<Json<BenchEntryImagePage>> {
    let worst = match p.sort.as_deref().map(str::trim) {
        None | Some("") | Some("order") => false,
        Some("worst") => true,
        Some(other) => {
            return Err(bad_query(format!("unknown sort '{other}'")).hint("sort is order or worst"));
        }
    };
    let limit = p.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT);
    Ok(Json(blocking(&st, move |s| s.benches()?.entry_images(&id, &entry_id, worst, p.offset, limit)).await?))
}

#[derive(Deserialize)]
struct StopEntryParams {
    #[serde(default)]
    only_if_open: bool,
    #[serde(default)]
    delete: bool,
}

async fn stop_entry(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
    ApiQuery(p): ApiQuery<StopEntryParams>,
) -> Result<Json<BenchEntry>> {
    if p.delete && !p.only_if_open {
        return Err(bad_query("delete=true requires only_if_open=true"));
    }
    Ok(Json(
        blocking(&st, move |s| {
            let benches = s.benches()?;
            if p.only_if_open && p.delete {
                benches.discard_open_entry(&id, &entry_id)
            } else if p.only_if_open {
                benches.stop_open_entry(&id, &entry_id)
            } else {
                benches.stop_entry(&id, &entry_id)
            }
        })
        .await?,
    ))
}

/// `POST …/entries/{entry_id}/predictions`: a part of an open external entry (up to 64 MiB).
async fn add_predictions(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
    ApiJson(req): ApiJson<BenchPredictions>,
) -> Result<Json<BenchEntry>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.add_predictions(&id, &entry_id, req.predictions)).await?))
}

/// `POST …/entries/{entry_id}/finish`: scores an open external entry.
async fn finish_entry(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
) -> Result<Json<BenchEntry>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.finish_open(&id, &entry_id)).await?))
}

async fn loops(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Json<BenchLoopList>> {
    Ok(Json(BenchLoopList { loops: st.store.benches()?.loops(&id)? }))
}

async fn start_loop(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<StartBenchLoop>,
) -> Result<Response> {
    // The loop drives the store from its own thread, holding it weakly.
    let weak = Arc::downgrade(&st.store);
    let lp: BenchLoop = blocking(&st, move |s| s.benches()?.start_loop(s, weak, &id, req)).await?;
    Ok((StatusCode::CREATED, Json(lp)).into_response())
}

async fn get_loop(
    State(st): State<AppState>,
    ApiPath((id, loop_id)): ApiPath<(String, String)>,
) -> Result<Json<BenchLoop>> {
    Ok(Json(st.store.benches()?.get_loop(&id, &loop_id)?))
}

async fn delete_loop(
    State(st): State<AppState>,
    ApiPath((id, loop_id)): ApiPath<(String, String)>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.benches()?.delete_loop(&id, &loop_id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

async fn stop_loop(
    State(st): State<AppState>,
    ApiPath((id, loop_id)): ApiPath<(String, String)>,
) -> Result<Json<BenchLoop>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.stop_loop(&id, &loop_id)).await?))
}

fn bench_sse_frames<F>(
    initial: SseFrame,
    rx: broadcast::Receiver<SseFrame>,
    resync: F,
) -> impl Stream<Item = SseFrame>
where
    F: Fn() -> Option<(SseFrame, broadcast::Receiver<SseFrame>)> + Send + 'static,
{
    stream::unfold((Some(initial), rx, resync), |(mut initial, mut rx, resync)| async move {
        if let Some(frame) = initial.take() {
            return Some((frame, (initial, rx, resync)));
        }
        match rx.recv().await {
            Ok(frame) => Some((frame, (initial, rx, resync))),
            Err(RecvError::Closed) => None,
            Err(RecvError::Lagged(_)) => {
                // Atomically snapshot the benchmark and advance to a tail-positioned receiver so stale buffered
                // frames never overwrite the resync snapshot or race with the client's reload.
                let (bench_frame, new_rx) = resync()?;
                rx = new_rx;
                Some((bench_frame, (initial, rx, resync)))
            }
        }
    })
}

/// `GET /api/bench/benchmarks/{id}/events`: a `bench` event, then live pre-serialized events until the client
/// disconnects, the benchmark is deleted or the server shuts down. A subscriber that falls behind gets the current
/// `bench` snapshot from the live tail.
async fn events(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let (initial, rx, slot) = st.store.benches()?.subscribe(&id)?;
    let mut shutdown = st.shutdown.subscribe();
    let stopped = async move {
        let _ = shutdown.wait_for(|v| *v).await;
    };
    let events = bench_sse_frames(initial, rx, move || Benches::resync(&slot))
        .map(|frame| Ok(frame.to_sse()))
        .take_until(stopped);
    Ok(Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive")))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::Store as _;
    use folo_api::{BenchEvent, BenchPlan, BenchRemoved, BenchSource};
    use tokio::sync::{Notify, broadcast};

    use super::*;

    fn sample_bench(name: &str, entries: u64) -> Bench {
        Bench {
            id: "bn_1".into(),
            name: name.into(),
            description: String::new(),
            classes: vec!["cat".into()],
            created: "2026-01-01T00:00:00Z".into(),
            source: BenchSource::LabelJob { job_id: "lb_1".into() },
            source_name: "src".into(),
            images: 4,
            boxes: 8,
            class_boxes: vec![8],
            plan: BenchPlan::default(),
            seed: 1,
            fingerprint: "fp".into(),
            entries,
            best: None,
            active: false,
        }
    }

    #[tokio::test]
    async fn bench_sse_lag_resyncs_to_live_tail_without_stale_buffered_frames() {
        let (tx, rx1) = broadcast::channel::<SseFrame>(2);
        let rx2 = tx.subscribe();
        let initial = SseFrame::bench(&BenchEvent::Bench(Box::new(sample_bench("before", 0))));
        let _ = tx.send(SseFrame::bench(&BenchEvent::Bench(Box::new(sample_bench("mid", 1)))));
        let _ = tx.send(SseFrame::bench(&BenchEvent::Removed(BenchRemoved { id: "be_1".into() })));
        let _ = tx.send(SseFrame::bench(&BenchEvent::Removed(BenchRemoved { id: "be_2".into() })));
        let _ = tx.send(SseFrame::bench(&BenchEvent::Bench(Box::new(sample_bench("stale", 2)))));

        let resynced = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(Notify::new());
        let tx_slot = Arc::new(std::sync::Mutex::new(Some(tx.clone())));
        let make_resync = || {
            let tx_slot = tx_slot.clone();
            let resynced = resynced.clone();
            let ready = ready.clone();
            move || {
                let sub = tx_slot.lock().unwrap().as_ref()?.subscribe();
                if resynced.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                    ready.notify_one();
                }
                Some((SseFrame::bench(&BenchEvent::Bench(Box::new(sample_bench("after", 3)))), sub))
            }
        };

        let post = tokio::spawn({
            let tx_slot = tx_slot.clone();
            let ready = ready.clone();
            async move {
                ready.notified().await;
                if let Some(tx) = tx_slot.lock().unwrap().take() {
                    let _ = tx
                        .send(SseFrame::bench(&BenchEvent::Removed(BenchRemoved { id: "be_fresh".into() })));
                }
            }
        });
        drop(tx);

        let (frames1, frames2): (Vec<SseFrame>, Vec<SseFrame>) = tokio::join!(
            bench_sse_frames(initial.clone(), rx1, make_resync()).collect(),
            bench_sse_frames(initial, rx2, make_resync()).collect(),
        );
        post.await.unwrap();

        assert_eq!(
            frames1.iter().map(|f| f.event).collect::<Vec<_>>(),
            ["bench", "bench", "removed"],
            "stale buffered bench/removed frames must not follow the resync snapshot"
        );
        let resync: Bench = serde_json::from_str(&frames1[1].data).unwrap();
        assert_eq!((resync.name.as_str(), resync.entries), ("after", 3));
        assert!(frames1[2].data.contains("be_fresh"));
        assert!(Arc::ptr_eq(&frames1[2].data, &frames2[2].data));

        // Deletion while lagged closes the stream cleanly without emitting stale buffered frames.
        let (tx_del, rx_del) = broadcast::channel::<SseFrame>(1);
        let _ = tx_del.send(SseFrame::new("removed", "{}"));
        let _ = tx_del.send(SseFrame::new("removed", "{}"));
        drop(tx_del);
        let del: Vec<SseFrame> = bench_sse_frames(
            SseFrame::bench(&BenchEvent::Bench(Box::new(sample_bench("before", 0)))),
            rx_del,
            || None,
        )
        .collect()
        .await;
        assert_eq!(del.iter().map(|f| f.event).collect::<Vec<_>>(), ["bench"]);
    }

    #[tokio::test]
    async fn benches_subscribe_shares_frames_and_closes_on_delete() {
        let dir = folo_test_support::TestDir::new("bench-sse-shared");
        let store = crate::RealStore::open(dir.path().to_path_buf()).device("cpu");
        let ds = store
            .create_synthetic(folo_api::SyntheticDataset {
                name: "bds".into(),
                images: 4,
                classes: vec!["cat".into()],
                imgsz: 64,
                seed: 1,
            })
            .unwrap();
        let benches = store.benches().unwrap();
        let bench = benches
            .create(
                &store,
                CreateBench {
                    name: "b1".into(),
                    description: String::new(),
                    source: BenchSource::Dataset { dataset_id: ds.id, split: Some(folo_api::Split::Val) },
                    max_images: Some(4),
                    seed: 1,
                    plan: None,
                },
            )
            .unwrap();

        let (init, mut rx1, weak) = benches.subscribe(&bench.id).unwrap();
        let (_, mut rx2, _) = benches.subscribe(&bench.id).unwrap();
        assert_eq!(init.event, "bench");
        let (snap, mut rx3) = Benches::resync(&weak).expect("alive bench resyncs");
        assert_eq!(snap.event, "bench");

        benches
            .update(&bench.id, UpdateBench { name: Some("b2".into()), description: None, plan: None })
            .unwrap();
        let f1 = rx1.recv().await.unwrap();
        let f2 = rx2.recv().await.unwrap();
        let f3 = rx3.recv().await.unwrap();
        assert_eq!(f1.event, "bench");
        assert!(f1.data.contains("b2"));
        assert!(Arc::ptr_eq(&f1.data, &f2.data), "concurrent subscribers share one Arc<str>");
        assert!(Arc::ptr_eq(&f1.data, &f3.data), "resynced subscriber shares the same Arc<str>");

        benches.delete(&bench.id).unwrap();
        assert!(Benches::resync(&weak).is_none());
        assert!(matches!(rx1.recv().await, Err(broadcast::error::RecvError::Closed)));
        assert!(matches!(rx2.recv().await, Err(broadcast::error::RecvError::Closed)));
        assert!(matches!(rx3.recv().await, Err(broadcast::error::RecvError::Closed)));
        store.shutdown();
    }
}
