//! `/api/uploads`: upload sessions in parts.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use folo_api::{CreateUpload, Upload, Uploaded};

use crate::AppState;
use crate::error::{ApiBytes, ApiJson, ApiPath, Result};

use super::blocking;

// ---- uploads -----------------------------------------------------------------------------------------------------

/// `POST /api/uploads`: `201` with a new session, or `200` with what an upload of the same fingerprint became.
pub(super) async fn create_upload(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<CreateUpload>,
) -> Result<Response> {
    let created = blocking(&st, move |s| s.create_upload(req)).await?;
    let status = if created.upload.is_some() { StatusCode::CREATED } else { StatusCode::OK };
    Ok((status, Json(created)).into_response())
}

pub(super) async fn get_upload(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<Upload>> {
    Ok(Json(blocking(&st, move |s| s.get_upload(&id)).await?))
}

/// `PUT /api/uploads/{id}/parts/{n}`: the body is the next bytes of the stream (at most
/// [`MAX_PART_BYTES`](crate::uploads::MAX_PART_BYTES)).
pub(super) async fn upload_part(
    State(st): State<AppState>,
    ApiPath((id, n)): ApiPath<(String, u32)>,
    ApiBytes(bytes): ApiBytes,
) -> Result<Json<Upload>> {
    Ok(Json(blocking(&st, move |s| s.upload_part(&id, n, bytes)).await?))
}

pub(super) async fn complete_upload(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Response> {
    let done: Uploaded = blocking(&st, move |s| s.complete_upload(&id)).await?;
    Ok((StatusCode::CREATED, Json(done)).into_response())
}

pub(super) async fn delete_upload(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.delete_upload(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}
