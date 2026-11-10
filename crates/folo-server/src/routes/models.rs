//! `/api/models` and `/api/val`: models, their evaluation and export, and weights downloads (conditional and
//! range requests).

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use folo_api::{ExportSpec, ModelList, OnnxSwitches, ValReport, ValRequest};
use serde::Deserialize;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{WeightsBody, WeightsFormat};

use super::{WEIGHTS_CHUNK, blocking};

// ---- models ------------------------------------------------------------------------------------------------------

pub(super) async fn list_models(State(st): State<AppState>) -> Result<Json<ModelList>> {
    Ok(Json(ModelList { models: blocking(&st, |s| s.list_models()).await? }))
}

pub(super) async fn get_model(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.get_model(&id)).await?).into_response())
}

pub(super) async fn model_eval(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.model_eval(&id)).await?).into_response())
}

pub(super) async fn export_model(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(spec): ApiJson<ExportSpec>,
) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.export_model(&id, spec)).await?).into_response())
}

#[derive(Deserialize)]
pub(super) struct WeightsParams {
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
pub(super) async fn model_weights(
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

pub(super) async fn val(
    State(st): State<AppState>,
    ApiJson(req): ApiJson<ValRequest>,
) -> Result<Json<ValReport>> {
    Ok(Json(blocking(&st, move |s| s.val(req)).await?))
}
