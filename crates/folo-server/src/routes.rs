//! HTTP handlers for every endpoint of `docs/API.md`.

use std::convert::Infallible;
use std::time::Duration;

use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::extract::{DefaultBodyLimit, FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use folo_api::{
    API_VERSION, DatasetList, ExportRequest, Health, ImagePage, IssuePage, LogLevel, LogLine, ModelList,
    PredictRequest, RegisterDataset, RunEvent, RunList, RunLogs, Split, SyntheticDataset, SystemInfo,
    TrainConfig, ValReport, ValRequest,
};
use futures_util::{Stream, StreamExt, stream};
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{ImageQuery, IssueQuery, PredictInput, PredictSource, Store, WeightsBody, WeightsFormat};
use crate::util::{rfc3339, unix_now};

/// Largest accepted request body for `POST /api/predict` uploads.
pub const MAX_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Default and maximum page size of `GET /api/datasets/{id}/images`.
pub const DEFAULT_LIMIT: u64 = 100;
/// Maximum page size of `GET /api/datasets/{id}/images`.
pub const MAX_LIMIT: u64 = 1000;
/// SSE keep-alive comment interval.
pub const KEEP_ALIVE: Duration = Duration::from_secs(15);
/// Chunk size of streamed weights files.
pub const WEIGHTS_CHUNK: usize = 256 * 1024;

/// The `/api` router.
pub fn api() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/system", get(system))
        .route("/datasets", get(list_datasets).post(register_dataset))
        .route("/datasets/synthetic", post(create_synthetic))
        .route("/datasets/{id}", get(get_dataset).delete(delete_dataset))
        .route("/datasets/{id}/images", get(list_images))
        .route("/datasets/{id}/issues", get(list_issues))
        .route("/datasets/{id}/images/{image_id}", get(image))
        .route("/models", get(list_models))
        .route("/models/{id}", get(get_model))
        .route("/models/{id}/eval", get(model_eval))
        .route("/models/{id}/export", post(export_model))
        .route("/models/{id}/weights", get(model_weights))
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/{id}", get(get_run).delete(delete_run))
        .route("/runs/{id}/stop", post(stop_run))
        .route("/runs/{id}/metrics", get(run_metrics))
        .route("/runs/{id}/samples", get(run_samples))
        .route("/runs/{id}/logs", get(run_logs))
        .route("/runs/{id}/events", get(run_events))
        .route("/predict", post(predict).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)))
        .route("/predict/stream", get(predict_stream))
        .route("/val", post(val))
        .fallback(api_not_found)
        .method_not_allowed_fallback(method_not_allowed)
}

/// Runs a store call on the blocking pool (stores may read files or render images).
async fn blocking<T, F>(st: &AppState, f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce(&dyn Store) -> Result<T> + Send + 'static,
{
    let store = st.store.clone();
    tokio::task::spawn_blocking(move || f(&*store))
        .await
        .map_err(|e| Error::internal(format!("task failed: {e}")))?
}

async fn api_not_found(req: Request) -> Error {
    Error::new(
        StatusCode::NOT_FOUND,
        "route_not_found",
        format!("no route for {} {}", req.method(), req.uri().path()),
    )
    .hint("see docs/API.md for the endpoints")
}

async fn method_not_allowed(req: Request) -> Error {
    Error::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        format!("{} is not allowed on {}", req.method(), req.uri().path()),
    )
}

// ---- system ------------------------------------------------------------------------------------------------------

async fn health(State(st): State<AppState>) -> Json<Health> {
    Json(Health {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        api: API_VERSION.into(),
        mock: st.store.is_mock(),
    })
}

async fn system(State(st): State<AppState>) -> Result<Json<SystemInfo>> {
    let probe = st.probe.clone();
    let (cpu, memory) = tokio::task::spawn_blocking(move || probe.sample())
        .await
        .map_err(|e| Error::internal(format!("task failed: {e}")))?;
    Ok(Json(SystemInfo { cpu, memory, gpus: st.store.gpus() }))
}

// ---- datasets ----------------------------------------------------------------------------------------------------

async fn list_datasets(State(st): State<AppState>) -> Result<Json<DatasetList>> {
    Ok(Json(DatasetList { datasets: st.store.list_datasets()? }))
}

async fn register_dataset(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<RegisterDataset>,
) -> Result<Response> {
    let ds = blocking(&st, move |s| s.register_dataset(req)).await?;
    Ok((StatusCode::CREATED, Json(ds)).into_response())
}

async fn create_synthetic(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<SyntheticDataset>,
) -> Result<Response> {
    let ds = blocking(&st, move |s| s.create_synthetic(req)).await?;
    Ok((StatusCode::CREATED, Json(ds)).into_response())
}

async fn get_dataset(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.get_dataset(&id)).await?).into_response())
}

async fn delete_dataset(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<StatusCode> {
    blocking(&st, move |s| s.delete_dataset(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct ImagesParams {
    split: Option<Split>,
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
    class: Option<u32>,
}

async fn list_images(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<ImagesParams>,
) -> Result<Json<ImagePage>> {
    let q = ImageQuery {
        split: p.split,
        offset: p.offset,
        limit: p.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT),
        class: p.class,
    };
    Ok(Json(blocking(&st, move |s| s.list_images(&id, &q)).await?))
}

#[derive(Deserialize)]
struct IssuesParams {
    kind: Option<String>,
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
}

async fn list_issues(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<IssuesParams>,
) -> Result<Json<IssuePage>> {
    if let Some(k) = &p.kind
        && !folo_api::ISSUE_KINDS.contains(&k.as_str())
    {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_query",
            format!("unknown issue kind '{k}'"),
        )
        .hint(format!("kind is one of {}", folo_api::ISSUE_KINDS.join(", "))));
    }
    let q =
        IssueQuery { kind: p.kind, offset: p.offset, limit: p.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT) };
    Ok(Json(blocking(&st, move |s| s.list_issues(&id, &q)).await?))
}

#[derive(Deserialize)]
struct ThumbParams {
    thumb: Option<u32>,
}

async fn image(
    State(st): State<AppState>,
    ApiPath((id, image_id)): ApiPath<(String, String)>,
    ApiQuery(p): ApiQuery<ThumbParams>,
) -> Result<Response> {
    let thumb = p.thumb.map(|t| t.clamp(16, 2048));
    let img = blocking(&st, move |s| s.image(&id, &image_id, thumb)).await?;
    let cache = if img.immutable { "public, max-age=31536000, immutable" } else { "no-cache" };
    Ok((
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(img.content_type)),
            (header::CACHE_CONTROL, HeaderValue::from_static(cache)),
        ],
        img.bytes,
    )
        .into_response())
}

// ---- models ------------------------------------------------------------------------------------------------------

async fn list_models(State(st): State<AppState>) -> Result<Json<ModelList>> {
    Ok(Json(ModelList { models: blocking(&st, |s| s.list_models()).await? }))
}

async fn get_model(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.get_model(&id)).await?).into_response())
}

async fn model_eval(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.model_eval(&id)).await?).into_response())
}

async fn export_model(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<ExportRequest>,
) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.export_model(&id, req.format)).await?).into_response())
}

#[derive(Deserialize)]
struct WeightsParams {
    format: Option<String>,
}

/// True when the conditional headers say the client's copy is current (RFC 9110 §13.2.2 for GET / HEAD):
/// `If-None-Match` decides when present; otherwise `If-Modified-Since` (one valid HTTP-date) against
/// `modified`, at the one-second precision of HTTP dates.
fn not_modified(headers: &HeaderMap, etag: &str, modified: std::time::SystemTime) -> bool {
    if let Some(m) = crate::web::if_none_match(headers, etag) {
        return m;
    }
    let mut ims = headers.get_all(header::IF_MODIFIED_SINCE).iter();
    let (Some(v), None) = (ims.next(), ims.next()) else { return false };
    let Some(since) = v.to_str().ok().and_then(|v| httpdate::parse_http_date(v).ok()) else { return false };
    let secs = |t: std::time::SystemTime| t.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    secs(modified) <= secs(since)
}

/// `GET /api/models/{id}/weights[?format=f16]`: the `.folo` file, streamed, with `ETag`, `Last-Modified` and
/// `304 Not Modified` for a current `If-None-Match` / `If-Modified-Since`. The compression layer leaves it alone
/// (see [`crate::Server`]), so `Content-Length` and the strong ETag describe the bytes on the wire.
async fn model_weights(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<WeightsParams>,
    headers: HeaderMap,
) -> Result<Response> {
    let format = match p.format.as_deref() {
        None | Some("") => WeightsFormat::Stored,
        Some("f16") => WeightsFormat::F16,
        Some(f) => {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_query",
                format!("unknown format '{f}'"),
            )
            .hint("format is f16, or leave it out for the stored file"));
        }
    };
    let w = blocking(&st, move |s| s.model_weights(&id, format)).await?;
    let etag = HeaderValue::from_str(&w.etag).map_err(|e| Error::internal(format!("bad ETag: {e}")))?;
    let validators = [
        (header::ETAG, etag),
        (header::LAST_MODIFIED, HeaderValue::from_str(&httpdate::fmt_http_date(w.modified)).expect("ASCII")),
        // Models change (a run's last.folo every epoch): caches revalidate with the validators above.
        (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
    ];
    if not_modified(&headers, &w.etag, w.modified) {
        return Ok((StatusCode::NOT_MODIFIED, validators).into_response());
    }
    let disposition = format!("attachment; filename=\"{}\"", w.file_name);
    let body = match w.body {
        WeightsBody::Bytes(b) => Body::from(b),
        WeightsBody::File(f) => {
            // At most `len` bytes, the length announced in Content-Length even if the file grew meanwhile.
            let file = tokio::io::AsyncReadExt::take(tokio::fs::File::from_std(f), w.len);
            Body::from_stream(tokio_util::io::ReaderStream::with_capacity(file, WEIGHTS_CHUNK))
        }
    };
    Ok((
        validators,
        [
            (header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream")),
            (header::CONTENT_LENGTH, HeaderValue::from(w.len)),
            (header::CONTENT_DISPOSITION, HeaderValue::from_str(&disposition).expect("sanitised name")),
        ],
        body,
    )
        .into_response())
}

async fn val(State(st): State<AppState>, ApiJson(req): ApiJson<ValRequest>) -> Result<Json<ValReport>> {
    Ok(Json(blocking(&st, move |s| s.val(req)).await?))
}

// ---- runs --------------------------------------------------------------------------------------------------------

async fn list_runs(State(st): State<AppState>) -> Result<Json<RunList>> {
    Ok(Json(RunList { runs: st.store.list_runs()? }))
}

fn body_error(e: impl std::fmt::Display) -> Error {
    Error::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
        format!("Failed to deserialize the JSON body into the target type: {e}"),
    )
}

/// `POST /api/runs`: a `TrainConfig` starts a new run (`201`); with `resume_run_id` it resumes that run instead
/// (`200`, same id).
async fn create_run(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Response> {
    let serde_json::Value::Object(fields) = body else {
        return Err(body_error("expected a JSON object"));
    };
    if let Some(rid) = fields.get("resume_run_id").filter(|v| !v.is_null()) {
        let rid = rid.as_str().ok_or_else(|| body_error("resume_run_id: expected a string"))?.to_string();
        let run = blocking(&st, move |s| s.resume_run(&rid, &fields)).await?;
        return Ok(Json(run).into_response());
    }
    let config: TrainConfig =
        serde_json::from_value(serde_json::Value::Object(fields)).map_err(body_error)?;
    let run = blocking(&st, move |s| s.create_run(config)).await?;
    Ok((StatusCode::CREATED, Json(run)).into_response())
}

async fn get_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(st.store.get_run(&id)?).into_response())
}

async fn stop_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.stop_run(&id)).await?).into_response())
}

async fn delete_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<StatusCode> {
    blocking(&st, move |s| s.delete_run(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct MetricsParams {
    every: Option<u64>,
}

async fn run_metrics(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<MetricsParams>,
) -> Result<Response> {
    let every = p.every.unwrap_or(1).max(1);
    Ok(Json(blocking(&st, move |s| s.run_metrics(&id, every)).await?).into_response())
}

#[derive(Deserialize)]
struct SamplesParams {
    epoch: Option<u32>,
}

async fn run_samples(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<SamplesParams>,
) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.run_samples(&id, p.epoch)).await?).into_response())
}

/// Default number of lines of `GET /api/runs/{id}/logs`.
pub const DEFAULT_LOG_LINES: usize = 500;
/// Maximum number of lines of `GET /api/runs/{id}/logs`.
pub const MAX_LOG_LINES: usize = 5000;

#[derive(Deserialize)]
struct LogsParams {
    limit: Option<usize>,
}

async fn run_logs(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<LogsParams>,
) -> Result<Json<RunLogs>> {
    let limit = p.limit.unwrap_or(DEFAULT_LOG_LINES).clamp(1, MAX_LOG_LINES);
    Ok(Json(RunLogs { lines: blocking(&st, move |s| s.run_logs(&id, limit)).await? }))
}

fn sse_event(ev: &RunEvent) -> Event {
    Event::default().event(ev.name()).data(ev.data_json())
}

/// `GET /api/runs/{id}/events`: a `status` event, then live events until the client disconnects, the run is
/// deleted or the server shuts down. Each client owns only a broadcast receiver; dropping the response stream drops
/// it, and the run's event pump stops by itself once no receivers are left.
async fn run_events(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let sub = st.store.subscribe(&id)?;
    let mut shutdown = st.shutdown.subscribe();
    let first = stream::once(std::future::ready(RunEvent::Status(sub.initial)));
    let rest = BroadcastStream::new(sub.events).map(|r| match r {
        Ok(ev) => ev,
        Err(BroadcastStreamRecvError::Lagged(n)) => RunEvent::Log(LogLine {
            t: rfc3339(unix_now()),
            level: LogLevel::Warn,
            message: format!("client fell behind; {n} events dropped (reload /metrics for the full history)"),
        }),
    });
    let stopped = async move {
        let _ = shutdown.wait_for(|v| *v).await;
    };
    let events = first.chain(rest).map(|ev| Ok(sse_event(&ev))).take_until(stopped);
    Ok(Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive")))
}

// ---- inference ---------------------------------------------------------------------------------------------------

fn field_err(name: &str, e: impl std::fmt::Display) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_body", format!("field `{name}`: {e}"))
}

async fn parse_multipart(mut mp: Multipart) -> Result<PredictInput> {
    let mut image = None;
    let (mut model_id, mut dataset_id, mut image_id) = (None, None, None);
    let (mut conf, mut iou, mut max_det) =
        (folo_api::default_conf(), folo_api::default_iou(), folo_api::default_max_det());
    while let Some(field) = mp.next_field().await.map_err(|e| Error::multipart("multipart", e))? {
        let name = field.name().unwrap_or("").to_string();
        if name == "image" {
            image = Some(field.bytes().await.map_err(|e| Error::multipart("image", e))?.to_vec());
            continue;
        }
        let text = field.text().await.map_err(|e| Error::multipart(&name, e))?;
        let text = text.trim().to_string();
        match name.as_str() {
            "model_id" => model_id = Some(text),
            "dataset_id" => dataset_id = Some(text),
            "image_id" => image_id = Some(text),
            "conf" => conf = text.parse().map_err(|e| field_err("conf", e))?,
            "iou" => iou = text.parse().map_err(|e| field_err("iou", e))?,
            "max_det" => max_det = text.parse().map_err(|e| field_err("max_det", e))?,
            _ => {}
        }
    }
    let model_id = model_id.ok_or_else(|| field_err("model_id", "missing"))?;
    let source = match (image, dataset_id, image_id) {
        (Some(bytes), _, _) if !bytes.is_empty() => PredictSource::Upload(bytes),
        (_, Some(dataset_id), Some(image_id)) => PredictSource::Dataset { dataset_id, image_id },
        _ => return Err(field_err("image", "missing or empty (send a file, or dataset_id + image_id)")),
    };
    Ok(PredictInput { model_id, source, conf, iou, max_det })
}

fn check_thresholds(p: &PredictInput) -> Result<()> {
    if !(0.0..=1.0).contains(&p.conf) {
        return Err(field_err("conf", "must be in 0..=1"));
    }
    if !(0.0..=1.0).contains(&p.iou) {
        return Err(field_err("iou", "must be in 0..=1"));
    }
    if p.max_det == 0 {
        return Err(field_err("max_det", "must be at least 1"));
    }
    Ok(())
}

async fn predict(State(st): State<AppState>, req: Request) -> Result<Response> {
    let ct = req
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let input = if ct.starts_with("multipart/form-data") {
        let mp = Multipart::from_request(req, &st).await?;
        parse_multipart(mp).await?
    } else {
        let ApiJson(r) = ApiJson::<PredictRequest>::from_request(req, &st).await?;
        PredictInput {
            model_id: r.model_id,
            source: PredictSource::Dataset { dataset_id: r.dataset_id, image_id: r.image_id },
            conf: r.conf,
            iou: r.iou,
            max_det: r.max_det,
        }
    };
    check_thresholds(&input)?;
    Ok(Json(blocking(&st, move |s| s.predict(input)).await?).into_response())
}

async fn predict_stream() -> Error {
    Error::not_implemented("streaming inference over WebSocket")
        .hint("use POST /api/predict per frame for now")
}
