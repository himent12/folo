//! Integration tests: every endpoint of docs/API.md in mock mode through `tower::ServiceExt::oneshot`, plus SSE,
//! the real store, static files, CORS and compression.

mod datasets;
mod harness;
mod http;
mod metrics;
mod models;
mod runs;
mod system;
