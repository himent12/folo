//! `/api/label`: labeling jobs, their images, runs, logs and live events, and exported datasets.

use std::convert::Infallible;

use axum::Json;
use axum::extract::{FromRequest, Multipart, Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use folo_api::{
    CreateLabelJob, ImportLabelImages, LabelAgentList, LabelDatasetRequest, LabelEvent, LabelImage,
    LabelImagePage, LabelImageStatus, LabelImagesAdded, LabelJob, LabelJobList, LabelReviewResult,
    ReviewLabelImages, RunLogs, StartLabeling, UpdateLabelImage, UpdateLabelJob,
};
use futures_util::{Stream, StreamExt, stream};
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};

use super::datasets::serve_image;
use super::inference::{field_err, is_multipart};
use super::{
    DEFAULT_LIMIT, DEFAULT_LOG_LINES, KEEP_ALIVE, LogsParams, MAX_LIMIT, MAX_LOG_LINES, ThumbParams, blocking,
};

// ---- agent labeling ------------------------------------------------------------------------------------------------

pub(super) async fn label_agents(State(st): State<AppState>) -> Result<Json<LabelAgentList>> {
    // Probing runs each CLI's `--version`.
    Ok(Json(LabelAgentList { agents: blocking(&st, |s| Ok(s.labeler()?.agents())).await? }))
}

pub(super) async fn label_jobs(State(st): State<AppState>) -> Result<Json<LabelJobList>> {
    Ok(Json(LabelJobList { jobs: st.store.labeler()?.list() }))
}

pub(super) async fn create_label_job(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<CreateLabelJob>,
) -> Result<Response> {
    let job = blocking(&st, move |s| s.labeler()?.create(req)).await?;
    Ok((StatusCode::CREATED, Json(job)).into_response())
}

pub(super) async fn label_job(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<LabelJob>> {
    Ok(Json(st.store.labeler()?.get(&id)?))
}

pub(super) async fn update_label_job(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<UpdateLabelJob>,
) -> Result<Json<LabelJob>> {
    Ok(Json(blocking(&st, move |s| s.labeler()?.update(&id, req)).await?))
}

pub(super) async fn delete_label_job(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.labeler()?.delete(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

/// `POST /api/label/jobs/{id}/images`: multipart uploads (one `image` part per file), or JSON naming a folder on
/// the server's machine.
pub(super) async fn add_label_images(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    req: Request,
) -> Result<Json<LabelImagesAdded>> {
    st.store.labeler()?.get(&id)?;
    if !is_multipart(&req) {
        let ApiJson(body) = ApiJson::<ImportLabelImages>::from_request(req, &st).await?;
        return Ok(Json(blocking(&st, move |s| s.labeler()?.add_folder(&id, &body.path)).await?));
    }
    let mut mp = Multipart::from_request(req, &st).await?;
    let mut uploads = Vec::new();
    while let Some(field) = mp.next_field().await.map_err(|e| Error::multipart("multipart", e))? {
        if field.name() != Some("image") {
            continue;
        }
        let name = field.file_name().unwrap_or("image").to_string();
        let bytes = field.bytes().await.map_err(|e| Error::multipart("image", e))?.to_vec();
        uploads.push(crate::label::Upload { name, bytes });
    }
    if uploads.is_empty() {
        return Err(field_err("image", "missing (send one `image` file part per image)"));
    }
    Ok(Json(blocking(&st, move |s| s.labeler()?.add_uploads(&id, uploads)).await?))
}

#[derive(Deserialize)]
pub(super) struct LabelImagesParams {
    status: Option<String>,
    reviewed: Option<bool>,
    /// Comma-separated image ids.
    ids: Option<String>,
    /// An absolute path on the server's machine (only compared, never stored).
    path: Option<String>,
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
}

pub(super) async fn label_images(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<LabelImagesParams>,
) -> Result<Json<LabelImagePage>> {
    let status = match p.status.as_deref() {
        None => None,
        Some(s) => Some(LabelImageStatus::parse(s).ok_or_else(|| {
            Error::new(StatusCode::BAD_REQUEST, "invalid_query", format!("unknown image status '{s}'"))
                .hint("status is one of pending, labeling, labeled, failed")
        })?),
    };
    let ids =
        p.ids.map(|ids| ids.split(',').map(str::trim).filter(|i| !i.is_empty()).map(String::from).collect());
    let q = crate::label::LabelImageQuery {
        status,
        reviewed: p.reviewed,
        ids,
        path: p.path.map(std::path::PathBuf::from),
        offset: p.offset,
        limit: p.limit.unwrap_or(DEFAULT_LIMIT).min(MAX_LIMIT),
    };
    if q.path.is_some() {
        // Resolving `path` and every image's file touches the disk.
        return Ok(Json(blocking(&st, move |s| s.labeler()?.query(&id, &q)).await?));
    }
    Ok(Json(st.store.labeler()?.query(&id, &q)?))
}

pub(super) async fn label_image(
    State(st): State<AppState>,
    ApiPath((id, image_id)): ApiPath<(String, String)>,
    ApiQuery(p): ApiQuery<ThumbParams>,
    headers: HeaderMap,
) -> Result<Response> {
    let thumb = p.thumb.map(|t| t.clamp(16, 2048));
    serve_image(&st, &headers, move |s| s.labeler()?.image(&id, &image_id, thumb)).await
}

pub(super) async fn update_label_image(
    State(st): State<AppState>,
    ApiPath((id, image_id)): ApiPath<(String, String)>,
    ApiJson(req): ApiJson<UpdateLabelImage>,
) -> Result<Json<LabelImage>> {
    Ok(Json(blocking(&st, move |s| s.labeler()?.update_image(&id, &image_id, req)).await?))
}

pub(super) async fn review_label_images(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<ReviewLabelImages>,
) -> Result<Json<LabelReviewResult>> {
    Ok(Json(blocking(&st, move |s| s.labeler()?.review(&id, &req)).await?))
}

pub(super) async fn delete_label_image(
    State(st): State<AppState>,
    ApiPath((id, image_id)): ApiPath<(String, String)>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.labeler()?.delete_image(&id, &image_id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

pub(super) async fn start_labeling(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<StartLabeling>,
) -> Result<Json<LabelJob>> {
    Ok(Json(blocking(&st, move |s| s.labeler()?.start(&id, req)).await?))
}

pub(super) async fn stop_labeling(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<LabelJob>> {
    Ok(Json(st.store.labeler()?.stop(&id)?))
}

pub(super) async fn label_logs(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<LogsParams>,
) -> Result<Json<RunLogs>> {
    let limit = p.limit.unwrap_or(DEFAULT_LOG_LINES).clamp(1, MAX_LOG_LINES);
    // A labeling job's log has no cursor: `total` stays 0, which is left out of the answer.
    Ok(Json(RunLogs { lines: st.store.labeler()?.logs(&id, limit)?, total: 0 }))
}

/// `GET /api/label/jobs/{id}/events`: a `job` event, then live events until the client disconnects, the job is
/// deleted or the server shuts down. A subscriber that falls behind is told to `reload`.
pub(super) async fn label_events(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let (job, events) = st.store.labeler()?.subscribe(&id)?;
    let mut shutdown = st.shutdown.subscribe();
    let first = stream::once(std::future::ready(LabelEvent::Job(Box::new(job))));
    let rest = BroadcastStream::new(events).map(|r| match r {
        Ok(ev) => ev,
        Err(BroadcastStreamRecvError::Lagged(_)) => LabelEvent::Reload,
    });
    let stopped = async move {
        let _ = shutdown.wait_for(|v| *v).await;
    };
    let events = first
        .chain(rest)
        .map(|ev| Ok(Event::default().event(ev.name()).data(ev.data_json())))
        .take_until(stopped);
    Ok(Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive")))
}

pub(super) async fn label_dataset(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<LabelDatasetRequest>,
) -> Result<Response> {
    let ds = blocking(&st, move |s| s.label_dataset(&id, req)).await?;
    Ok((StatusCode::CREATED, Json(ds)).into_response())
}
