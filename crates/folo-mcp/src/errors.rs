//! Tool errors. They reuse the HTTP API's error type (status + `{ code, message, hint }`), so a store error and the
//! tool error an agent sees carry the same code. The status only matters in proxy mode, where errors come from HTTP.

use http::StatusCode;

pub use folo_server::error::Error;

/// Tool arguments that do not match the tool's `inputSchema` (missing or wrongly typed fields, bad values).
pub fn invalid_arguments(message: impl Into<String>) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_arguments", message)
        .hint("check the tool's inputSchema (tools/list)")
}

/// A mutating tool called on a `--read-only` server.
pub fn read_only(tool: &str) -> Error {
    Error::new(
        StatusCode::FORBIDDEN,
        "read_only",
        format!("`{tool}` changes state and this server is read-only"),
    )
    .hint("restart `folo mcp` without --read-only to allow it")
}

/// A tool the current backend cannot serve yet.
pub fn not_implemented(what: &str, hint: impl Into<String>) -> Error {
    Error::new(StatusCode::NOT_IMPLEMENTED, "not_implemented", format!("{what} is not implemented yet"))
        .hint(hint)
}
