//! The error envelope (`{ "error": { code, message, hint } }`) and extractors that use it for rejections.

use axum::Json;
use axum::extract::multipart::{MultipartError, MultipartRejection};
use axum::extract::rejection::{JsonRejection, PathRejection, QueryRejection};
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

    /// `404 <what>_not_found`, e.g. `not_found("dataset", "ds_1")`.
    pub fn not_found(what: &str, id: &str) -> Error {
        let code = format!("{what}_not_found");
        Error::new(StatusCode::NOT_FOUND, &code, format!("{what} '{id}' does not exist"))
    }

    /// `409 conflict`-style error with a specific code.
    pub fn conflict(code: &str, message: impl Into<String>) -> Error {
        Error::new(StatusCode::CONFLICT, code, message)
    }

    /// `501 not_implemented`.
    pub fn not_implemented(what: &str) -> Error {
        Error::new(StatusCode::NOT_IMPLEMENTED, "not_implemented", format!("{what} is not implemented yet"))
            .hint("the FOLO engine is not wired into the server yet; run `folo serve --mock` for mock data")
    }

    /// `500 internal`.
    pub fn internal(message: impl Into<String>) -> Error {
        Error::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", message)
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

/// `Path<T>` whose rejection is our error envelope.
#[derive(FromRequestParts)]
#[from_request(via(axum::extract::Path), rejection(Error))]
pub struct ApiPath<T>(pub T);

/// `Query<T>` whose rejection is our error envelope.
#[derive(FromRequestParts)]
#[from_request(via(axum::extract::Query), rejection(Error))]
pub struct ApiQuery<T>(pub T);
