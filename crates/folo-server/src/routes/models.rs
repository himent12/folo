//! `/api/models` and `/api/val`: models, their evaluation and export, and weights downloads (conditional and
//! range requests).

use axum::Json;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use folo_api::{ExportSpec, ModelList, OnnxSwitches, ValReport, ValRequest};
use serde::{Deserialize, Serialize};

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{
    DEFAULT_VAL_DIAG_BG_IOU, DEFAULT_VAL_DIAG_CONF, DEFAULT_VAL_DIAG_TP_IOU, DEFAULT_VAL_DIAG_WORST,
    DynamicAxes, ValDiagConfig, ValDiagnostics, WeightsBody, WeightsFormat,
};

use super::{WEIGHTS_CHUNK, blocking};

// ---- models ------------------------------------------------------------------------------------------------------

/// Weak ETag (`W/"<32 hex>"`, 128-bit SHA-256 of canonical JSON bytes) for compressible JSON model endpoints.
/// Weak because [`crate::Server`]'s compression layer may gzip-encode `application/json`, and RFC 9110 §8.8.3.3
/// forbids sharing a strong validator across different content codings.
fn json_etag(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(36);
    out.push_str("W/\"");
    for b in &digest[..16] {
        let _ = write!(&mut out, "{b:02x}");
    }
    out.push('"');
    out
}

/// Serializes `value` once, derives its weak `ETag`, and answers `412 Precondition Failed` (`If-Match`),
/// `304 Not Modified` (`If-None-Match`), or `200 OK` with `ETag`, `Cache-Control: no-cache`, and
/// `Vary: accept-encoding`.
fn cached_json_response(value: &impl Serialize, headers: &HeaderMap) -> Result<Response> {
    let bytes =
        serde_json::to_vec(value).map_err(|e| Error::internal(format!("serializing JSON response: {e}")))?;
    let etag = json_etag(&bytes);
    let etag_val = HeaderValue::from_str(&etag).expect("ASCII ETag");
    if !crate::web::if_match(headers, &etag) {
        return Ok((
            StatusCode::PRECONDITION_FAILED,
            [
                (header::ETAG, etag_val),
                (header::CACHE_CONTROL, HeaderValue::from_static("no-store")),
                (header::VARY, HeaderValue::from_static("accept-encoding")),
            ],
        )
            .into_response());
    }
    let common = [
        (header::ETAG, etag_val),
        (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
        (header::VARY, HeaderValue::from_static("accept-encoding")),
    ];
    if crate::web::if_none_match(headers, &etag) == Some(true) {
        return Ok((StatusCode::NOT_MODIFIED, common).into_response());
    }
    Ok((
        StatusCode::OK,
        common,
        [(header::CONTENT_TYPE, HeaderValue::from_static("application/json"))],
        bytes,
    )
        .into_response())
}

pub(super) async fn list_models(State(st): State<AppState>, headers: HeaderMap) -> Result<Response> {
    let list = ModelList { models: blocking(&st, |s| s.list_models()).await? };
    cached_json_response(&list, &headers)
}

pub(super) async fn get_model(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    headers: HeaderMap,
) -> Result<Response> {
    let (model, arch) = blocking(&st, move |s| s.get_model_with_arch(&id)).await?;
    let mut res = cached_json_response(&model, &headers)?;
    if let Ok(val) = HeaderValue::from_str(&arch) {
        res.headers_mut().insert(axum::http::HeaderName::from_static("x-folo-arch"), val);
    }
    Ok(res)
}

pub(super) async fn model_eval(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    headers: HeaderMap,
) -> Result<Response> {
    let eval = blocking(&st, move |s| s.model_eval(&id)).await?;
    cached_json_response(&eval, &headers)
}

#[derive(Deserialize)]
pub(super) struct ExportModelBody {
    #[serde(flatten)]
    base: ExportSpec,
    /// ONNX only: symbolic batch axis (`"batch"`).
    #[serde(default)]
    dynamic_batch: Option<bool>,
    /// ONNX only: symbolic spatial axes (`"height"`, `"width"`).
    #[serde(default)]
    dynamic_hw: Option<bool>,
}

pub(super) async fn export_model(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(body): ApiJson<ExportModelBody>,
) -> Result<Response> {
    let spec = body.base;
    if spec.format != folo_api::ExportFormat::Onnx
        && (body.dynamic_batch.is_some() || body.dynamic_hw.is_some())
    {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_body",
            "dynamic_batch and dynamic_hw only apply to format `onnx`",
        ));
    }
    if spec.onnx.dynamic && (body.dynamic_batch == Some(false) || body.dynamic_hw == Some(false)) {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_body",
            "`dynamic: true` cannot be combined with `dynamic_batch: false` or `dynamic_hw: false`",
        ));
    }
    let axes = DynamicAxes {
        batch: spec.onnx.dynamic || body.dynamic_batch.unwrap_or(false),
        hw: spec.onnx.dynamic || body.dynamic_hw.unwrap_or(false),
    };
    Ok(Json(blocking(&st, move |s| s.export_model_with_axes(&id, spec, axes)).await?).into_response())
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
    if !crate::web::if_match(&headers, &w.etag) {
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

#[derive(Deserialize)]
pub(super) struct ValBody {
    #[serde(flatten)]
    base: ValRequest,
    #[serde(default)]
    diagnostics: bool,
    #[serde(default)]
    worst: Option<usize>,
    #[serde(default)]
    diag_conf: Option<f32>,
    #[serde(default)]
    diag_tp_iou: Option<f32>,
    #[serde(default)]
    diag_bg_iou: Option<f32>,
}

#[derive(Serialize)]
pub(super) struct ValResponse {
    #[serde(flatten)]
    report: ValReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    diagnostics: Option<ValDiagnostics>,
}

pub(super) async fn val(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<ValBody>,
) -> Result<Json<ValResponse>> {
    let enabled = body.diagnostics || body.worst.is_some();
    if !enabled && (body.diag_conf.is_some() || body.diag_tp_iou.is_some() || body.diag_bg_iou.is_some()) {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_body",
            "diag_conf, diag_tp_iou and diag_bg_iou require diagnostics: true or worst",
        ));
    }
    let diag = enabled.then(|| ValDiagConfig {
        worst: body.worst.unwrap_or(DEFAULT_VAL_DIAG_WORST),
        conf: body.diag_conf.unwrap_or(DEFAULT_VAL_DIAG_CONF),
        tp_iou: body.diag_tp_iou.unwrap_or(DEFAULT_VAL_DIAG_TP_IOU),
        bg_iou: body.diag_bg_iou.unwrap_or(DEFAULT_VAL_DIAG_BG_IOU),
    });
    let (report, diagnostics) = blocking(&st, move |s| s.val_with_diagnostics(body.base, diag)).await?;
    Ok(Json(ValResponse { report, diagnostics }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_etag_is_weak_and_content_addressed() {
        let a = json_etag(br#"{"models":[]}"#);
        let b = json_etag(br#"{"models":[]}"#);
        let c = json_etag(br#"{"models":[1]}"#);
        assert!(a.starts_with("W/\"") && a.ends_with('"') && a.len() == 36, "{a}");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[tokio::test]
    async fn cached_json_response_handles_conditional_headers() {
        let payload = json!({ "models": [{ "id": "m_1" }] });
        let res = cached_json_response(&payload, &HeaderMap::new()).unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let etag = res.headers()[header::ETAG].to_str().unwrap().to_string();
        assert!(etag.starts_with("W/\""), "{etag}");
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
        assert_eq!(res.headers()[header::VARY], "accept-encoding");
        assert_eq!(res.headers()[header::CONTENT_TYPE], "application/json");
        let body = axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap();
        assert_eq!(serde_json::from_slice::<serde_json::Value>(&body).unwrap(), payload);

        let strong_form = etag.strip_prefix("W/").unwrap().to_string();
        for inm in [etag.clone(), strong_form, format!("\"other\", {etag}"), "*".into()] {
            let mut h = HeaderMap::new();
            h.insert(header::IF_NONE_MATCH, HeaderValue::from_str(&inm).unwrap());
            let res = cached_json_response(&payload, &h).unwrap();
            assert_eq!(res.status(), StatusCode::NOT_MODIFIED, "{inm}");
            assert_eq!(res.headers()[header::ETAG], etag.as_str());
            assert_eq!(res.headers()[header::CACHE_CONTROL], "no-cache");
            assert_eq!(res.headers()[header::VARY], "accept-encoding");
            assert!(axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap().is_empty());
        }

        let mut stale = HeaderMap::new();
        stale.insert(header::IF_NONE_MATCH, HeaderValue::from_static("W/\"stale\""));
        let res = cached_json_response(&payload, &stale).unwrap();
        assert_eq!(res.status(), StatusCode::OK);

        // RFC 9110 §13.1.1: weak ETag fails strong If-Match (412, never cached), while `*` passes.
        let mut im = HeaderMap::new();
        im.insert(header::IF_MATCH, HeaderValue::from_str(&etag).unwrap());
        let res = cached_json_response(&payload, &im).unwrap();
        assert_eq!(res.status(), StatusCode::PRECONDITION_FAILED);
        assert_eq!(res.headers()[header::CACHE_CONTROL], "no-store");
        assert!(axum::body::to_bytes(res.into_body(), usize::MAX).await.unwrap().is_empty());

        let mut im_star = HeaderMap::new();
        im_star.insert(header::IF_MATCH, HeaderValue::from_static("*"));
        let res = cached_json_response(&payload, &im_star).unwrap();
        assert_eq!(res.status(), StatusCode::OK);
    }
}
