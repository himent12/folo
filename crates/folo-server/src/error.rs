//! The error envelope (`{ "error": { code, message, hint } }`) and extractors that use it for rejections.

use axum::Json;
use axum::extract::multipart::{MultipartError, MultipartRejection};
use axum::extract::rejection::{BytesRejection, JsonRejection, PathRejection, QueryRejection};
use axum::extract::{FromRequest, FromRequestParts};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use folo_api::{ApiError, ErrorBody};

/// An API error: an HTTP status plus the JSON envelope.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    /// HTTP status.
    pub status: StatusCode,
    /// Body.
    pub body: ApiError,
}

/// Result alias for store and handler code.
pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    /// A new error.
    pub fn new(status: StatusCode, code: &str, message: impl Into<String>) -> Error {
        Error { status, body: ApiError { code: code.into(), message: message.into(), hint: None } }
    }

    /// Adds a hint.
    pub fn hint(mut self, hint: impl Into<String>) -> Error {
        self.body.hint = Some(hint.into());
        self
    }

    /// `400 bad_request`.
    pub fn bad_request(message: impl Into<String>) -> Error {
        Error::new(StatusCode::BAD_REQUEST, "bad_request", message)
    }

    /// `404 <what>_not_found`, e.g. `not_found("dataset", "ds_1")`, with a hint naming where the ones that exist
    /// are listed.
    pub fn not_found(what: &str, id: &str) -> Error {
        let code = format!("{what}_not_found");
        let hint = match what {
            "image" => {
                "list_dataset_images (MCP) or GET /api/datasets/{id}/images lists a dataset's images".into()
            }
            "label_job" => {
                "list_label_jobs (MCP), `folo label jobs` or GET /api/label/jobs lists the labeling jobs that exist"
                    .into()
            }
            _ => format!("list_{what}s (MCP) or GET /api/{what}s lists the {what}s that exist"),
        };
        let noun = what.replace('_', " ");
        Error::new(StatusCode::NOT_FOUND, &code, format!("{noun} '{id}' does not exist")).hint(hint)
    }

    /// `409 conflict`-style error with a specific code.
    pub fn conflict(code: &str, message: impl Into<String>) -> Error {
        Error::new(StatusCode::CONFLICT, code, message)
    }

    /// `409 run_not_active`: stopping a run that is not queued or running.
    pub fn run_not_active(id: &str, status: &str) -> Error {
        Error::conflict("run_not_active", format!("run '{id}' is already {status}"))
            .hint("nothing to stop: the run has finished; run_status (GET /api/runs/{id}) has its result")
    }

    /// `501 not_implemented`.
    pub fn not_implemented(what: &str) -> Error {
        Error::new(StatusCode::NOT_IMPLEMENTED, "not_implemented", format!("{what} is not implemented yet"))
            .hint("this server cannot do that yet; see docs/API.md for what each mode supports")
    }

    /// `500 internal`.
    pub fn internal(message: impl Into<String>) -> Error {
        Error::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
    }

    /// `500 io_error`: a file could not be read or written (a full disk, permissions).
    pub fn io(message: impl Into<String>) -> Error {
        Error::new(StatusCode::INTERNAL_SERVER_ERROR, "io_error", message)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.status.as_u16(), self.body.code, self.body.message)
    }
}

impl std::error::Error for Error {}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        (self.status, Json(ErrorBody { error: self.body })).into_response()
    }
}

impl From<JsonRejection> for Error {
    fn from(r: JsonRejection) -> Error {
        let status = r.status();
        let code = match r {
            JsonRejection::MissingJsonContentType(_) => "unsupported_media_type",
            JsonRejection::BytesRejection(_) if status == StatusCode::PAYLOAD_TOO_LARGE => {
                "payload_too_large"
            }
            _ => "invalid_body",
        };
        Error::new(status, code, r.body_text())
    }
}

impl From<QueryRejection> for Error {
    fn from(r: QueryRejection) -> Error {
        Error::new(StatusCode::BAD_REQUEST, "invalid_query", r.body_text())
    }
}

impl From<PathRejection> for Error {
    fn from(r: PathRejection) -> Error {
        Error::new(StatusCode::BAD_REQUEST, "invalid_path", r.body_text())
    }
}

impl From<BytesRejection> for Error {
    fn from(r: BytesRejection) -> Error {
        let status = r.status();
        if status == StatusCode::PAYLOAD_TOO_LARGE {
            Error::new(status, "payload_too_large", r.body_text())
                .hint("send the stream in smaller parts: Upload.max_part_bytes is the largest part this server takes")
        } else {
            Error::new(StatusCode::BAD_REQUEST, "invalid_body", r.body_text())
        }
    }
}

/// Maps a multipart failure, keeping its status (`413 payload_too_large` when the body is over the limit).
fn multipart_error(status: StatusCode, what: &str, text: String) -> Error {
    if status == StatusCode::PAYLOAD_TOO_LARGE {
        Error::new(status, "payload_too_large", format!("{what}: {text}"))
            .hint("uploads are limited to 64 MiB")
    } else {
        let status = if status.is_client_error() { status } else { StatusCode::BAD_REQUEST };
        Error::new(status, "invalid_body", format!("{what}: {text}"))
    }
}

impl From<MultipartRejection> for Error {
    fn from(r: MultipartRejection) -> Error {
        multipart_error(r.status(), "multipart", r.body_text())
    }
}

impl Error {
    /// A multipart field failure.
    pub fn multipart(field: &str, e: MultipartError) -> Error {
        multipart_error(e.status(), &format!("field `{field}`"), e.body_text())
    }
}

/// `Json<T>` whose rejection is our error envelope.
#[derive(FromRequest)]
#[from_request(via(axum::Json), rejection(Error))]
pub struct ApiJson<T>(pub T);

/// Returns `true` iff `value` is a syntactically valid RFC 9110 media type whose type/subtype is
/// `application/octet-stream` (case-insensitive) with well-formed optional `; name=value` parameters.
fn is_octet_stream_media_type(value: &str) -> bool {
    let is_tchar = |b: u8| {
        b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'!' | b'#'
                    | b'$'
                    | b'%'
                    | b'&'
                    | b'\''
                    | b'*'
                    | b'+'
                    | b'-'
                    | b'.'
                    | b'^'
                    | b'_'
                    | b'`'
                    | b'|'
                    | b'~'
            )
    };
    let is_ows = |b: u8| b == b' ' || b == b'\t';
    let is_qdtext = |b: u8| b == b'\t' || (0x20..=0x7E).contains(&b) || b >= 0x80;
    let bytes = value.as_bytes();
    let mut i = 0;
    while i < bytes.len() && is_ows(bytes[i]) {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i] != b';' && !is_ows(bytes[i]) {
        i += 1;
    }
    if !value[start..i].eq_ignore_ascii_case("application/octet-stream") {
        return false;
    }
    loop {
        while i < bytes.len() && is_ows(bytes[i]) {
            i += 1;
        }
        if i == bytes.len() {
            return true;
        }
        if bytes[i] != b';' {
            return false;
        }
        i += 1;
        while i < bytes.len() && is_ows(bytes[i]) {
            i += 1;
        }
        let name_start = i;
        while i < bytes.len() && is_tchar(bytes[i]) {
            i += 1;
        }
        if i == name_start || i >= bytes.len() || bytes[i] != b'=' {
            return false;
        }
        i += 1;
        if i >= bytes.len() {
            return false;
        }
        if bytes[i] == b'"' {
            i += 1;
            let mut closed = false;
            while i < bytes.len() {
                match bytes[i] {
                    b'\\' => {
                        i += 1;
                        if i >= bytes.len() || !is_qdtext(bytes[i]) {
                            return false;
                        }
                        i += 1;
                    }
                    b'"' => {
                        i += 1;
                        closed = true;
                        break;
                    }
                    b if is_qdtext(b) => i += 1,
                    _ => return false,
                }
            }
            if !closed {
                return false;
            }
        } else {
            let val_start = i;
            while i < bytes.len() && is_tchar(bytes[i]) {
                i += 1;
            }
            if i == val_start {
                return false;
            }
        }
    }
}

/// A raw request body (an upload part) whose rejection is our error envelope.
///
/// Requires a single syntactically valid `Content-Type: application/octet-stream` header before reading the body,
/// returning `415 unsupported_media_type` when missing, duplicated, malformed, or mismatched.
pub struct ApiBytes(pub axum::body::Bytes);

impl<S: Send + Sync> FromRequest<S> for ApiBytes {
    type Rejection = Error;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<ApiBytes> {
        let mut headers = req.headers().get_all(axum::http::header::CONTENT_TYPE).iter();
        let ok_type = match (headers.next(), headers.next()) {
            (Some(v), None) => v.to_str().ok().is_some_and(is_octet_stream_media_type),
            _ => false,
        };
        if !ok_type {
            return Err(Error::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "expected request with `Content-Type: application/octet-stream`",
            ));
        }
        Ok(ApiBytes(axum::body::Bytes::from_request(req, state).await?))
    }
}

/// `Path<T>` whose rejection is our error envelope.
#[derive(FromRequestParts)]
#[from_request(via(axum::extract::Path), rejection(Error))]
pub struct ApiPath<T>(pub T);

/// `Query<T>` whose rejection is our error envelope.
#[derive(FromRequestParts)]
#[from_request(via(axum::extract::Query), rejection(Error))]
pub struct ApiQuery<T>(pub T);
