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
use axum::routing::{delete, get, post, put};
use folo_api::{
    API_VERSION, BenchmarkReport, BenchmarkRequest, CliCommand, CloudConnection, CloudConnectionList,
    CloudInstance, CloudInstanceList, CloudOfferList, CloudProviderList, ConnectCloud, CreateUpload,
    DatasetList, ExportSpec, Health, ImagePage, IssuePage, LogLevel, LogLine, ModelList, OnnxSwitches,
    PredictRequest, QueueRequest, RegisterDataset, RunEstimate, RunEvent, RunList, RunLogs, Split,
    SyntheticDataset, SystemInfo, SystemResponse, ThroughputTable, TrainConfig, Upload, Uploaded, ValReport,
    ValRequest,
};
use futures_util::{Stream, StreamExt, stream};
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use crate::AppState;
use crate::error::{ApiBytes, ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{
    ImageQuery, IssueQuery, LogsQuery, MetricsQuery, PredictInput, PredictSource, QueueMove, Store,
    WeightsBody, WeightsFormat,
};
use crate::uploads::MAX_PART_BYTES;
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
        .route("/system/throughput", get(throughput))
        .route("/system/benchmark", post(benchmark))
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
        .route("/runs/estimate", post(estimate_run))
        .route("/runs/{id}", get(get_run).delete(delete_run))
        .route("/runs/{id}/cli", get(run_cli))
        .route("/runs/{id}/stop", post(stop_run))
        .route("/runs/{id}/queue", post(move_run))
        .route("/runs/{id}/metrics", get(run_metrics))
        .route("/runs/{id}/samples", get(run_samples))
        .route("/runs/{id}/logs", get(run_logs))
        .route("/runs/{id}/events", get(run_events))
        .route("/predict", post(predict).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)))
        .route("/predict/stream", get(predict_stream))
        .route("/val", post(val))
        .route("/uploads", post(create_upload))
        .route("/uploads/{id}", get(get_upload).delete(delete_upload))
        .route("/uploads/{id}/parts/{n}", put(upload_part).layer(DefaultBodyLimit::max(MAX_PART_BYTES)))
        .route("/uploads/{id}/complete", post(complete_upload))
        .route("/cloud/providers", get(cloud_providers))
        .route("/cloud/connections", get(cloud_connections).post(connect_cloud))
        .route("/cloud/connections/{id}", delete(disconnect_cloud))
        .route("/cloud/connections/{id}/check", post(check_cloud))
        .route("/cloud/connections/{id}/offers", get(cloud_offers))
        .route("/cloud/instances", get(cloud_instances))
        .route("/cloud/instances/{id}", delete(terminate_instance))
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

async fn system(State(st): State<AppState>) -> Result<Json<SystemResponse>> {
    let probe = st.probe.clone();
    let store = st.store.clone();
    let ((cpu, memory), capabilities, gpu_pool) =
        tokio::task::spawn_blocking(move || (probe.sample(), store.capabilities(), store.gpu_pool()))
            .await
            .map_err(|e| Error::internal(format!("task failed: {e}")))?;
    let data_dir = st.store.data_dir().map(folo_api::json::path);
    let features: Vec<String> = [
        folo_api::FEATURE_EXPORT_ONNX_SWITCHES,
        folo_api::FEATURE_QUEUE_CONTROL,
        folo_api::FEATURE_WEIGHTS_ONNX,
        folo_api::FEATURE_THROUGHPUT,
        folo_api::FEATURE_RUN_ESTIMATE,
        folo_api::FEATURE_BENCHMARK,
        folo_api::FEATURE_RUN_CLI,
    ]
    .iter()
    .chain(st.store.features())
    .map(|f| f.to_string())
    .collect();
    let features = if st.web_wasm {
        features.into_iter().chain([folo_api::FEATURE_WEB_WASM.to_string()]).collect()
    } else {
        features
    };
    Ok(Json(SystemResponse {
        system: SystemInfo { cpu, memory, gpus: st.store.gpus() },
        data_dir,
        features,
        capabilities,
        gpu_pool,
    }))
}

async fn throughput(State(st): State<AppState>) -> Result<Json<ThroughputTable>> {
    Ok(Json(blocking(&st, |s| s.throughput()).await?))
}

async fn benchmark(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<BenchmarkRequest>,
) -> Result<Json<BenchmarkReport>> {
    Ok(Json(blocking(&st, move |s| s.benchmark(req)).await?))
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
    severity: Option<String>,
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
    let severity = match p.severity.as_deref() {
        None => None,
        Some(s) => Some(folo_api::IssueSeverity::parse(s).ok_or_else(|| {
            Error::new(StatusCode::BAD_REQUEST, "invalid_query", format!("unknown issue severity '{s}'"))
                .hint("severity is one of error, warning, info")
        })?),
    };
    let q = IssueQuery {
        kind: p.kind,
        severity,
        offset: p.offset,
        limit: p.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT),
    };
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
    ApiJson(spec): ApiJson<ExportSpec>,
) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.export_model(&id, spec)).await?).into_response())
}

#[derive(Deserialize)]
struct WeightsParams {
    format: Option<String>,
    /// `format=onnx` only: `folo export --decode`.
    decode: Option<bool>,
    /// `format=onnx` only: `folo export --dynamic`.
    dynamic: Option<bool>,
    /// `format=onnx` only: `folo export --half`.
    half: Option<bool>,
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

/// False when the request has `If-Match` and it names neither `etag` nor `*` (the strong comparison, RFC 9110
/// §13.1.1: a weak tag never matches).
fn if_match(headers: &HeaderMap, etag: &str) -> bool {
    let mut values = headers.get_all(header::IF_MATCH).iter().peekable();
    if values.peek().is_none() {
        return true;
    }
    values.filter_map(|v| v.to_str().ok()).any(|v| v.split(',').any(|t| t.trim() == "*" || t.trim() == etag))
}

/// What a request's `Range` header asks of a body of some length ([`byte_range`]).
#[derive(Clone, Debug, PartialEq, Eq)]
enum ByteRange {
    /// No header, or one this server does not understand (several ranges, a suffix range): the whole body.
    Whole,
    /// These bytes.
    Part(std::ops::Range<u64>),
    /// A range that starts at or past the end.
    Unsatisfiable,
}

/// The one range `bytes=<first>-<last>` or `bytes=<first>-` of the request's `Range` header, against a body of
/// `len` bytes (`<last>` clamped to it).
fn byte_range(headers: &HeaderMap, len: u64) -> ByteRange {
    let digits = |s: &str| {
        let s = s.trim();
        (!s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u64>().ok()).flatten()
    };
    let mut values = headers.get_all(header::RANGE).iter();
    let (Some(v), None) = (values.next(), values.next()) else { return ByteRange::Whole };
    let Some(spec) = v.to_str().ok().and_then(|v| v.trim().strip_prefix("bytes=")) else {
        return ByteRange::Whole;
    };
    let Some((first, last)) = spec.split_once('-') else { return ByteRange::Whole };
    let Some(first) = digits(first) else { return ByteRange::Whole };
    let end = match last.trim() {
        "" => len,
        last => match digits(last) {
            Some(last) if last >= first => last.saturating_add(1).min(len),
            _ => return ByteRange::Whole,
        },
    };
    if first >= len { ByteRange::Unsatisfiable } else { ByteRange::Part(first..end) }
}

/// `GET /api/models/{id}/weights[?format=f16]`: the `.folo` file, streamed, with `ETag`, `Last-Modified` and
/// `304 Not Modified` for a current `If-None-Match` / `If-Modified-Since`; `412` for an `If-Match` it fails; one
/// `Range` as `206` (or `416`). The compression layer leaves it alone (see [`crate::Server`]), so `Content-Length`
/// and the strong ETag describe the bytes on the wire.
async fn model_weights(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<WeightsParams>,
    headers: HeaderMap,
) -> Result<Response> {
    let bad = |m: String| Error::new(StatusCode::BAD_REQUEST, "invalid_query", m);
    let switches = OnnxSwitches {
        decode: p.decode.unwrap_or(false),
        dynamic: p.dynamic.unwrap_or(false),
        half: p.half.unwrap_or(false),
    };
    let format = match p.format.as_deref() {
        None | Some("") => WeightsFormat::Stored,
        Some("f16") => WeightsFormat::F16,
        Some("onnx") => WeightsFormat::Onnx(switches),
        Some(f) => {
            return Err(bad(format!("unknown format '{f}'"))
                .hint("format is f16 or onnx, or leave it out for the stored file"));
        }
    };
    if !matches!(format, WeightsFormat::Onnx(_))
        && (p.decode.is_some() || p.dynamic.is_some() || p.half.is_some())
    {
        return Err(bad("decode, dynamic and half only apply to format=onnx".into())
            .hint("format=f16 is already the half-precision .folo file; leave the switches out"));
    }
    let w = blocking(&st, move |s| s.model_weights(&id, format)).await?;
    let etag = HeaderValue::from_str(&w.etag).map_err(|e| Error::internal(format!("bad ETag: {e}")))?;
    let validators = [
        (header::ETAG, etag),
        (header::LAST_MODIFIED, HeaderValue::from_str(&httpdate::fmt_http_date(w.modified)).expect("ASCII")),
        // Models change (a run's last.folo every epoch): caches revalidate with the validators above.
        (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        (header::ACCEPT_RANGES, HeaderValue::from_static("bytes")),
    ];
    // A download in parts names the version it started with: another one is never mixed in.
    if !if_match(&headers, &w.etag) {
        return Ok((StatusCode::PRECONDITION_FAILED, validators).into_response());
    }
    if not_modified(&headers, &w.etag, w.modified) {
        return Ok((StatusCode::NOT_MODIFIED, validators).into_response());
    }
    let (status, range) = match byte_range(&headers, w.len) {
        ByteRange::Whole => (StatusCode::OK, 0..w.len),
        ByteRange::Part(r) => (StatusCode::PARTIAL_CONTENT, r),
        ByteRange::Unsatisfiable => {
            let all = HeaderValue::from_str(&format!("bytes */{}", w.len)).expect("ASCII");
            return Ok((StatusCode::RANGE_NOT_SATISFIABLE, validators, [(header::CONTENT_RANGE, all)])
                .into_response());
        }
    };
    let len = range.end - range.start;
    let body = match w.body {
        WeightsBody::Bytes(b) => Body::from(b.slice(range.start as usize..range.end as usize)),
        WeightsBody::File(f) => {
            let mut file = tokio::fs::File::from_std(f);
            if range.start > 0 {
                tokio::io::AsyncSeekExt::seek(&mut file, std::io::SeekFrom::Start(range.start))
                    .await
                    .map_err(|e| Error::internal(format!("seeking in {}: {e}", w.file_name)))?;
            }
            // At most the range's bytes, the length announced in Content-Length even if the file grew meanwhile.
            let file = tokio::io::AsyncReadExt::take(file, len);
            Body::from_stream(tokio_util::io::ReaderStream::with_capacity(file, WEIGHTS_CHUNK))
        }
    };
    let disposition = format!("attachment; filename=\"{}\"", w.file_name);
    let mut res = (
        status,
        validators,
        [
            (header::CONTENT_TYPE, HeaderValue::from_static("application/octet-stream")),
            (header::CONTENT_LENGTH, HeaderValue::from(len)),
            (header::CONTENT_DISPOSITION, HeaderValue::from_str(&disposition).expect("sanitised name")),
        ],
        body,
    )
        .into_response();
    if status == StatusCode::PARTIAL_CONTENT {
        let part = format!("bytes {}-{}/{}", range.start, range.end - 1, w.len);
        res.headers_mut().insert(header::CONTENT_RANGE, HeaderValue::from_str(&part).expect("ASCII"));
    }
    Ok(res)
}

async fn val(State(st): State<AppState>, ApiJson(req): ApiJson<ValRequest>) -> Result<Json<ValReport>> {
    Ok(Json(blocking(&st, move |s| s.val(req)).await?))
}

// ---- uploads -----------------------------------------------------------------------------------------------------

/// `POST /api/uploads`: `201` with a new session, or `200` with what an upload of the same fingerprint became.
async fn create_upload(State(st): State<AppState>, ApiJson(req): ApiJson<CreateUpload>) -> Result<Response> {
    let created = blocking(&st, move |s| s.create_upload(req)).await?;
    let status = if created.upload.is_some() { StatusCode::CREATED } else { StatusCode::OK };
    Ok((status, Json(created)).into_response())
}

async fn get_upload(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Json<Upload>> {
    Ok(Json(blocking(&st, move |s| s.get_upload(&id)).await?))
}

/// `PUT /api/uploads/{id}/parts/{n}`: the body is the next bytes of the stream (at most [`MAX_PART_BYTES`]).
async fn upload_part(
    State(st): State<AppState>,
    ApiPath((id, n)): ApiPath<(String, u32)>,
    ApiBytes(bytes): ApiBytes,
) -> Result<Json<Upload>> {
    Ok(Json(blocking(&st, move |s| s.upload_part(&id, n, bytes)).await?))
}

async fn complete_upload(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    let done: Uploaded = blocking(&st, move |s| s.complete_upload(&id)).await?;
    Ok((StatusCode::CREATED, Json(done)).into_response())
}

async fn delete_upload(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<StatusCode> {
    blocking(&st, move |s| s.delete_upload(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

// ---- cloud -------------------------------------------------------------------------------------------------------

async fn cloud_providers(State(st): State<AppState>) -> Result<Json<CloudProviderList>> {
    Ok(Json(CloudProviderList { providers: st.store.cloud_providers()? }))
}

async fn cloud_connections(State(st): State<AppState>) -> Result<Json<CloudConnectionList>> {
    Ok(Json(CloudConnectionList { connections: blocking(&st, |s| s.cloud_connections()).await? }))
}

/// `POST /api/cloud/connections`: checks the values with the provider, then stores them (`201`).
async fn connect_cloud(State(st): State<AppState>, ApiJson(req): ApiJson<ConnectCloud>) -> Result<Response> {
    let c = blocking(&st, move |s| s.connect_cloud(req)).await?;
    Ok((StatusCode::CREATED, Json(c)).into_response())
}

async fn check_cloud(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<CloudConnection>> {
    Ok(Json(blocking(&st, move |s| s.check_cloud(&id)).await?))
}

async fn disconnect_cloud(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<StatusCode> {
    blocking(&st, move |s| s.disconnect_cloud(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct OffersParams {
    #[serde(default)]
    refresh: bool,
}

async fn cloud_offers(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<OffersParams>,
) -> Result<Json<CloudOfferList>> {
    Ok(Json(blocking(&st, move |s| s.cloud_offers(&id, p.refresh)).await?))
}

async fn cloud_instances(State(st): State<AppState>) -> Result<Json<CloudInstanceList>> {
    Ok(Json(CloudInstanceList { instances: blocking(&st, |s| s.cloud_instances()).await? }))
}

async fn terminate_instance(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<CloudInstance>> {
    Ok(Json(blocking(&st, move |s| s.terminate_instance(&id)).await?))
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

/// A string field of a `POST /api/runs` body (`null` counts as missing).
fn string_field(fields: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<Option<String>> {
    match fields.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            Ok(Some(v.as_str().ok_or_else(|| body_error(format!("{key}: expected a string")))?.into()))
        }
    }
}

/// `POST /api/runs`: a `TrainConfig` starts a new run (`201`); with `resume_run_id` it resumes that run instead
/// (`200`, same id); with `checkpoint` the new run continues that model's training state.
async fn create_run(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Response> {
    let serde_json::Value::Object(mut fields) = body else {
        return Err(body_error("expected a JSON object"));
    };
    let checkpoint = string_field(&fields, "checkpoint")?;
    let resume = string_field(&fields, "resume_run_id")?;
    if checkpoint.is_some() {
        let with = if resume.is_some() {
            Some("resume_run_id")
        } else {
            string_field(&fields, "weights")?.map(|_| "weights")
        };
        if let Some(with) = with {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                format!("checkpoint cannot be combined with {with}"),
            )
            .hint(
                "checkpoint starts a new run from a checkpoint's training state; leave the other field out",
            ));
        }
    }
    fields.remove("checkpoint");
    if let Some(rid) = resume {
        let run = blocking(&st, move |s| s.resume_run(&rid, &fields)).await?;
        return Ok(Json(run).into_response());
    }
    let config: TrainConfig =
        serde_json::from_value(serde_json::Value::Object(fields)).map_err(body_error)?;
    let run = match checkpoint {
        Some(model) => blocking(&st, move |s| s.continue_run(&model, config)).await?,
        None => blocking(&st, move |s| s.create_run(config)).await?,
    };
    Ok((StatusCode::CREATED, Json(run)).into_response())
}

async fn estimate_run(
    State(st): State<AppState>,
    ApiJson(config): ApiJson<TrainConfig>,
) -> Result<Json<RunEstimate>> {
    Ok(Json(blocking(&st, move |s| s.estimate_run(config)).await?))
}

async fn get_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(st.store.get_run(&id)?).into_response())
}

async fn run_cli(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Json<CliCommand>> {
    Ok(Json(blocking(&st, move |s| s.run_cli(&id)).await?))
}

async fn stop_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.stop_run(&id)).await?).into_response())
}

/// `POST /api/runs/{id}/queue`: moves a waiting run in the training queue (or starts it next).
async fn move_run(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<QueueRequest>,
) -> Result<Response> {
    let mv = QueueMove::parse(req)?;
    Ok(Json(blocking(&st, move |s| s.move_run(&id, mv)).await?).into_response())
}

async fn delete_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<StatusCode> {
    blocking(&st, move |s| s.delete_run(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
struct MetricsParams {
    every: Option<u64>,
    after_step: Option<u64>,
    after_epoch: Option<u32>,
}

async fn run_metrics(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<MetricsParams>,
) -> Result<Response> {
    let q = MetricsQuery {
        every: p.every.unwrap_or(1).max(1),
        after_step: p.after_step,
        after_epoch: p.after_epoch,
    };
    Ok(Json(blocking(&st, move |s| s.run_metrics(&id, &q)).await?).into_response())
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
    after: Option<u64>,
}

async fn run_logs(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<LogsParams>,
) -> Result<Json<RunLogs>> {
    let q = LogsQuery { limit: p.limit.unwrap_or(DEFAULT_LOG_LINES).clamp(1, MAX_LOG_LINES), after: p.after };
    Ok(Json(blocking(&st, move |s| s.run_logs(&id, &q)).await?))
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
