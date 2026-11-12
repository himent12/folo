//! `/api/datasets`: registering, listing and deleting datasets, their images and issues.

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use folo_api::{DatasetList, ImagePage, IssuePage, RegisterDataset, Split, SyntheticDataset};
use serde::Deserialize;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{ImageData, ImageQuery, IssueQuery};

use super::{DEFAULT_LIMIT, MAX_LIMIT, ThumbParams, blocking};

// ---- datasets ----------------------------------------------------------------------------------------------------

pub(super) async fn list_datasets(State(st): State<AppState>) -> Result<Json<DatasetList>> {
    Ok(Json(DatasetList { datasets: st.store.list_datasets()? }))
}

pub(super) async fn register_dataset(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<RegisterDataset>,
) -> Result<Response> {
    let ds = blocking(&st, move |s| s.register_dataset(req)).await?;
    Ok((StatusCode::CREATED, Json(ds)).into_response())
}

pub(super) async fn create_synthetic(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<SyntheticDataset>,
) -> Result<Response> {
    let ds = blocking(&st, move |s| s.create_synthetic(req)).await?;
    Ok((StatusCode::CREATED, Json(ds)).into_response())
}

pub(super) async fn get_dataset(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.get_dataset(&id)).await?).into_response())
}

#[derive(Deserialize)]
pub(super) struct DeleteParams {
    /// Also delete the dataset's folder (only one the server wrote under `<data dir>/datasets`).
    #[serde(default)]
    files: bool,
}

pub(super) async fn delete_dataset(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<DeleteParams>,
) -> Result<StatusCode> {
    blocking(&st, move |s| if p.files { s.delete_dataset_files(&id) } else { s.delete_dataset(&id) }).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub(super) struct ImagesParams {
    split: Option<Split>,
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
    class: Option<u32>,
}

pub(super) async fn list_images(
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
pub(super) struct IssuesParams {
    kind: Option<String>,
    severity: Option<String>,
    #[serde(default)]
    offset: u64,
    limit: Option<u64>,
}

pub(super) async fn list_issues(
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

pub(super) async fn image(
    State(st): State<AppState>,
    ApiPath((id, image_id)): ApiPath<(String, String)>,
    ApiQuery(p): ApiQuery<ThumbParams>,
) -> Result<Response> {
    let thumb = p.thumb.map(|t| t.clamp(16, 2048));
    let img = blocking(&st, move |s| s.image(&id, &image_id, thumb)).await?;
    Ok(image_response(img))
}

/// An image as a response: its bytes with their type, cacheable forever when they never change.
pub(crate) fn image_response(img: ImageData) -> Response {
    let cache = if img.immutable { "public, max-age=31536000, immutable" } else { "no-cache" };
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(img.content_type)),
            (header::CACHE_CONTROL, HeaderValue::from_static(cache)),
        ],
        img.bytes,
    )
        .into_response()
}
