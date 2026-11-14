//! FOLO HTTP API types: the serde request and response bodies of `docs/API.md`.
//!
//! This crate holds data types only (no IO, no async), so it can be shared by `folo-server`, the CLI, `folo-client`,
//! the MCP server (`folo-mcp`, with the `schema` feature), `folo-cloud`, folo-vision's trainer (its config and
//! events), `folo-bench`, `folo-wasm` and FOLO Desktop (`apps/desktop`, a separate cargo workspace that depends on
//! this crate by path). Field names and enum spellings are the wire format; keep them in sync with `docs/API.md` and
//! `web/src/lib/api/types/` (re-exported by `types.ts`).
//!
//! Conventions: times are RFC 3339 strings, durations are seconds (`*_s`) or milliseconds (`*_ms`), boxes are pixel
//! `xyxy` in the original image, ids are opaque strings.
//!
//! Modules, each named for the `docs/API.md` section it models (all re-exported at the crate root, except the three
//! public ones):
//! - `system`: *System*; `datasets`: *Datasets*; `models`: *Models*; `inference`: *Inference*; `validation`:
//!   *Validation*;
//! - `runs`: *Training runs* (the config, a run, the queue); `run_events`: what a run reports (metrics, samples,
//!   logs, SSE events);
//! - `throughput`: *Throughput and estimates*; `label`: *Agent labeling*; `bench/`: *Vision benchmarks*;
//! - `cloud`: *Cloud compute* and *Uploads*; `errors`: *Errors* (the error body);
//! - [`codes`]: every error code with its HTTP status and CLI exit code; [`json`]: JSON presentation helpers of the
//!   CLI and MCP; [`weights`]: the published weights, their cache folder and FOLO's home folder.
//!
//! The `schema` feature derives `schemars::JsonSchema` for every type (off by default; `folo-mcp` turns it on).

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
