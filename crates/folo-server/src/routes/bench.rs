//! The `/api/bench` handlers (docs/API.md, *Vision benchmarks*): glue between HTTP and [`Benches`],
//! which every store that has benchmarks hands out ([`crate::Store::benches`]; the mock has none: `501`).

use std::convert::Infallible;
use std::sync::Arc;

use axum::Json;
use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use folo_api::{
    Bench, BenchCatalog, BenchDesign, BenchEntry, BenchEntryImagePage, BenchEntryList, BenchEvent,
    BenchImagePage, BenchList, BenchLoop, BenchLoopList, BenchPredictions, BenchResult, CreateBench,
    DesignBench, StartBenchEntry, StartBenchLoop, UpdateBench,
};
use folo_bench::Condition;
use futures_util::{Stream, StreamExt, stream};
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use crate::AppState;
use crate::bench::Benches;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::routes::{DEFAULT_LIMIT, KEEP_ALIVE, MAX_LIMIT, MAX_UPLOAD_BYTES, blocking, image_response};

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

async fn create(State(st): State<AppState>, ApiJson(req): ApiJson<CreateBench>) -> Result<Response> {
    let bench: Bench = blocking(&st, move |s| s.benches()?.create(s, req)).await?;
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
) -> Result<Response> {
    let v = variant(p.condition, p.severity)?;
    let thumb = p.thumb.map(|t| t.clamp(16, 2048));
    let img = blocking(&st, move |s| s.benches()?.image(&id, &image_id, v, thumb)).await?;
    Ok(image_response(img))
}

async fn entries(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Json<BenchEntryList>> {
    Ok(Json(BenchEntryList { entries: st.store.benches()?.entries(&id)? }))
}

async fn start_entry(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<StartBenchEntry>,
) -> Result<Response> {
    // An external entry is scored in the request: on the blocking pool, as everything else here.
    let entry: BenchEntry = blocking(&st, move |s| s.benches()?.start_entry(s, &id, req)).await?;
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

async fn stop_entry(
    State(st): State<AppState>,
    ApiPath((id, entry_id)): ApiPath<(String, String)>,
) -> Result<Json<BenchEntry>> {
    Ok(Json(blocking(&st, move |s| s.benches()?.stop_entry(&id, &entry_id)).await?))
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

/// `GET /api/bench/benchmarks/{id}/events`: a `bench` event, then live events until the client disconnects, the
/// benchmark is deleted or the server shuts down. A subscriber that falls behind gets the current `bench`.
async fn events(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let (bench, rx, slot) = st.store.benches()?.subscribe(&id)?;
    let mut shutdown = st.shutdown.subscribe();
    let first = stream::once(std::future::ready(Some(BenchEvent::Bench(Box::new(bench)))));
    let rest = BroadcastStream::new(rx).map(move |r| match r {
        Ok(ev) => Some(ev),
        Err(BroadcastStreamRecvError::Lagged(_)) => {
            Benches::current(&slot).map(|b| BenchEvent::Bench(Box::new(b)))
        }
    });
    let stopped = async move {
        let _ = shutdown.wait_for(|v| *v).await;
    };
    let events = first
        .chain(rest)
        .filter_map(std::future::ready)
        .map(|ev| Ok(Event::default().event(ev.name()).data(ev.data_json())))
        .take_until(stopped);
    Ok(Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive")))
}
