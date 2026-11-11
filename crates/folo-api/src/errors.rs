//! Errors (docs/API.md, *Errors*): the body of every non-2xx response.

use serde::{Deserialize, Serialize};

/// The body of every non-2xx response: `{ "error": { "code", "message", "hint" } }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ErrorBody {
    /// The error itself.
    pub error: ApiError,
}

/// A machine-readable error: a stable `code`, a human `message` and an optional `hint` on how to fix it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ApiError {
    /// Stable snake_case code, e.g. `dataset_not_found`. See the table in `docs/API.md`.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// What to do about it, if there is something useful to say. Always serialized (`null` when absent).
    #[serde(default)]
    pub hint: Option<String>,
}
