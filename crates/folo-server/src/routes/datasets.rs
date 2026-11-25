//! `/api/datasets`: registering, listing and deleting datasets, their images and issues.

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use folo_api::{DatasetList, ImagePage, IssuePage, RegisterDataset, Split, SyntheticDataset};
use serde::Deserialize;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{ImageData, ImageQuery, IssueQuery, Store};

use super::{DEFAULT_LIMIT, MAX_LIMIT, ThumbParams, blocking};

// ---- datasets ----------------------------------------------------------------------------------------------------

pub(super) async fn list_datasets(State(st): State<AppState>) -> Result<Json<DatasetList>> {
    let datasets = blocking(&st, |s| s.list_datasets()).await?;
    Ok(Json(DatasetList { datasets }))
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
    headers: HeaderMap,
) -> Result<Response> {
    let thumb = p.thumb.map(|t| t.clamp(16, 2048));
    serve_image(&st, &headers, move |s| s.image(&id, &image_id, thumb)).await
}

/// Strong ETag (`"<32 hex>"`, 128-bit SHA-256 of content type and bytes) for an image response.
/// The compression layer skips non-SVG `image/*` responses (`DefaultPredicate`), and every image endpoint serves
/// raster formats only (JPEG, PNG, BMP, WebP), so the strong validator describes the exact bytes on the wire.
fn image_etag(img: &ImageData) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    let mut h = Sha256::new();
    h.update(img.content_type.as_bytes());
    h.update([0]);
    h.update(&img.bytes);
    let digest = h.finalize();
    let mut out = String::with_capacity(34);
    out.push('"');
    for b in &digest[..16] {
        let _ = write!(&mut out, "{b:02x}");
    }
    out.push('"');
    out
}

/// Loads an image on the blocking pool, hashes it for a strong ETag there, and builds the response (answering
/// `304 Not Modified` when `headers`'s `If-None-Match` matches).
pub(crate) async fn serve_image<F>(st: &AppState, headers: &HeaderMap, load: F) -> Result<Response>
where
    F: FnOnce(&dyn Store) -> Result<ImageData> + Send + 'static,
{
    let (img, etag) = blocking(st, move |s| {
        let img = load(s)?;
        let etag = image_etag(&img);
        Ok((img, etag))
    })
    .await?;
    Ok(image_response(img, &etag, headers))
}

/// An image as a response: its bytes with their type, strong `ETag` and `Cache-Control`, `412 Precondition Failed`
/// when `If-Match` fails (RFC 9110 §13.2.2), or `304 Not Modified` when `If-None-Match` matches `etag`.
fn image_response(img: ImageData, etag: &str, headers: &HeaderMap) -> Response {
    let cache = if img.immutable { "public, max-age=31536000, immutable" } else { "no-cache" };
    let etag_val = HeaderValue::from_str(etag).expect("ASCII ETag");
    if !crate::web::if_match(headers, etag) {
        return (
            StatusCode::PRECONDITION_FAILED,
            [(header::ETAG, etag_val), (header::CACHE_CONTROL, HeaderValue::from_static("no-store"))],
        )
            .into_response();
    }
    if crate::web::if_none_match(headers, etag) == Some(true) {
        return (
            StatusCode::NOT_MODIFIED,
            [(header::ETAG, etag_val), (header::CACHE_CONTROL, HeaderValue::from_static(cache))],
        )
            .into_response();
    }
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(img.content_type)),
            (header::CACHE_CONTROL, HeaderValue::from_static(cache)),
            (header::ETAG, etag_val),
        ],
        img.bytes,
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_image(content_type: &'static str, bytes: &[u8], immutable: bool) -> ImageData {
        ImageData { content_type, bytes: bytes.to_vec(), immutable }
    }

    #[test]
    fn etag_is_strong_and_tracks_bytes_and_content_type() {
        let a = image_etag(&sample_image("image/jpeg", b"hello", false));
        let same = image_etag(&sample_image("image/jpeg", b"hello", true));
        let diff_bytes = image_etag(&sample_image("image/jpeg", b"world", false));
        let diff_type = image_etag(&sample_image("image/png", b"hello", false));
        assert!(a.starts_with('"') && a.ends_with('"') && !a.starts_with("W/"), "{a}");
        assert_eq!(a.len(), 34, "quoted 128-bit hex tag: {a}");
        assert_eq!(a, same);
        assert_ne!(a, diff_bytes);
        assert_ne!(a, diff_type);
    }

    #[tokio::test]
    async fn image_response_handles_conditional_if_none_match() {
        let img = sample_image("image/jpeg", b"jpeg-bytes", false);
        let etag = image_etag(&img);

        let res = image_response(img.clone(), &etag, &HeaderMap::new());
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::ETAG], etag.as_str());
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
        assert_eq!(res.headers()[header::CONTENT_TYPE], "image/jpeg");
        let body = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        assert_eq!(body.as_ref(), b"jpeg-bytes");

        for inm in [etag.clone(), format!("W/{etag}"), format!("\"other\", {etag}"), "*".into()] {
            let mut h = HeaderMap::new();
            h.insert(header::IF_NONE_MATCH, HeaderValue::from_str(&inm).unwrap());
            let res = image_response(img.clone(), &etag, &h);
            assert_eq!(res.status(), StatusCode::NOT_MODIFIED, "{inm}");
            assert_eq!(res.headers()[header::ETAG], etag.as_str());
            assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
            let body = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
            assert!(body.is_empty(), "{inm}");
        }

        let mut mismatch = HeaderMap::new();
        mismatch.insert(header::IF_NONE_MATCH, HeaderValue::from_static("\"stale\""));
        let res = image_response(sample_image("image/jpeg", b"jpeg-bytes", true), &etag, &mismatch);
        assert_eq!(res.status(), StatusCode::OK);
        assert_eq!(res.headers()[header::CACHE_CONTROL], "public, max-age=31536000, immutable");

        // RFC 9110 §13.2.2: If-Match (strong) takes precedence over If-None-Match, and 412 is never cached.
        let mut both = HeaderMap::new();
        both.insert(header::IF_MATCH, HeaderValue::from_static("\"stale\""));
        both.insert(header::IF_NONE_MATCH, HeaderValue::from_static("*"));
        let res = image_response(sample_image("image/jpeg", b"jpeg-bytes", true), &etag, &both);
        assert_eq!(res.status(), StatusCode::PRECONDITION_FAILED);
        assert_eq!(res.headers()[header::ETAG], etag.as_str());
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
        assert!(axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap().is_empty());
    }
}
