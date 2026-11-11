//! FOLO HTTP API types: the serde request and response bodies of `docs/API.md`.
//!
//! This crate holds data types only (no IO, no async), so it can be shared by `folo-server`, the CLI and FOLO
//! Desktop (`apps/desktop`, a separate cargo workspace that depends on this crate by path). Field names and enum
//! spellings are the wire format; keep them in sync with `docs/API.md` and `web/src/lib/api/types/` (re-exported by
//! `types.ts`).
//!
//! Conventions: times are RFC 3339 strings, durations are seconds (`*_s`) or milliseconds (`*_ms`), boxes are pixel
//! `xyxy` in the original image, ids are opaque strings.

mod bench;
mod cloud;
pub mod codes;
mod datasets;
mod errors;
mod inference;
pub mod json;
mod label;
mod models;
mod run_events;
mod runs;
mod system;
mod throughput;
mod validation;
pub mod weights;

pub use bench::*;
pub use cloud::*;
pub use datasets::*;
pub use errors::*;
pub use inference::*;
pub use label::*;
pub use models::*;
pub use run_events::*;
pub use runs::*;
pub use system::*;
pub use throughput::*;
pub use validation::*;

#[cfg(test)]
mod tests;
