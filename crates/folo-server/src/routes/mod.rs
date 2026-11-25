//! HTTP handlers for every endpoint of `docs/API.md`.
//!
//! [`api`] is the `/api` router: every path and the handler it calls. A handler parses the request, calls the
//! [`Store`] (through [`blocking`] when the call may read files or burn CPU) and turns the answer into JSON, SSE or
//! bytes; errors leave as [`Error`]. One file per area, named for the `docs/API.md` section it serves:
//! - `system`: *System* (`/health`, `/system`) and *Throughput and estimates* (`/system/throughput`,
//!   `/system/benchmark`);
//! - `datasets`: *Datasets*; `models`: *Models* (`/models`) and *Validation* (`/val`); `inference`: *Inference*;
//! - `runs`: *Training runs* (also `/runs/estimate` of *Throughput and estimates*), with the SSE stream of a run;
//! - `uploads`: *Uploads*; `cloud`: *Cloud compute*; `label`: *Agent labeling*;
//! - `bench`: *Vision benchmarks* (its own [`Router`], merged into [`api`]).

mod bench;
mod cloud;
mod datasets;
mod inference;
mod label;
mod models;
mod runs;
pub(crate) mod system;
mod uploads;

use std::time::Duration;

use axum::Router;
use axum::extract::{DefaultBodyLimit, Request};
use axum::http::StatusCode;
use axum::routing::{delete, get, post, put};
use serde::Deserialize;

use crate::AppState;
use crate::error::{Error, Result};
use crate::store::Store;
use crate::uploads::MAX_PART_BYTES;
use cloud::{
    check_cloud, cloud_connections, cloud_instances, cloud_offers, cloud_providers, connect_cloud,
    disconnect_cloud, terminate_instance,
};
use datasets::{
    create_synthetic, delete_dataset, get_dataset, image, list_datasets, list_images, list_issues,
    register_dataset,
};
use inference::{predict, predict_stream};
use label::{
    add_label_images, create_label_job, delete_label_image, delete_label_job, label_agents, label_dataset,
    label_events, label_image, label_images, label_job, label_jobs, label_logs, review_label_images,
    start_labeling, stop_labeling, update_label_image, update_label_job,
};
use models::{export_model, get_model, list_models, model_eval, model_weights, val};
use runs::{
    create_run, delete_run, estimate_run, get_run, list_runs, move_run, run_cli, run_events, run_logs,
    run_metrics, run_samples, stop_run,
};
use system::{benchmark, health, system, throughput};
use uploads::{complete_upload, create_upload, delete_upload, get_upload, upload_part};

pub(crate) use datasets::serve_image;

/// Largest accepted request body for uploads (`POST /api/predict`, images of a labeling job).
pub const MAX_UPLOAD_BYTES: usize = 64 * 1024 * 1024;
/// Default and maximum page size of `GET /api/datasets/{id}/images`.
pub const DEFAULT_LIMIT: u64 = 100;
/// Maximum page size of `GET /api/datasets/{id}/images`.
pub const MAX_LIMIT: u64 = 1000;
/// SSE keep-alive comment interval.
pub const KEEP_ALIVE: Duration = Duration::from_secs(15);
/// Chunk size of streamed weights files.
pub const WEIGHTS_CHUNK: usize = 256 * 1024;

/// The `/api` router.
pub fn api() -> Router<AppState> {
    Router::new()
        .route("/health", get(health))
        .route("/system", get(system))
        .route("/system/throughput", get(throughput))
        .route("/system/benchmark", post(benchmark))
        .route("/datasets", get(list_datasets).post(register_dataset))
        .route("/datasets/synthetic", post(create_synthetic))
        .route("/datasets/{id}", get(get_dataset).delete(delete_dataset))
        .route("/datasets/{id}/images", get(list_images))
        .route("/datasets/{id}/issues", get(list_issues))
        .route("/datasets/{id}/images/{image_id}", get(image))
        .route("/models", get(list_models))
        .route("/models/{id}", get(get_model))
        .route("/models/{id}/eval", get(model_eval))
        .route("/models/{id}/export", post(export_model))
        .route("/models/{id}/weights", get(model_weights))
        .route("/runs", get(list_runs).post(create_run))
        .route("/runs/estimate", post(estimate_run))
        .route("/runs/{id}", get(get_run).delete(delete_run))
        .route("/runs/{id}/cli", get(run_cli))
        .route("/runs/{id}/stop", post(stop_run))
        .route("/runs/{id}/queue", post(move_run))
        .route("/runs/{id}/metrics", get(run_metrics))
        .route("/runs/{id}/samples", get(run_samples))
        .route("/runs/{id}/logs", get(run_logs))
        .route("/runs/{id}/events", get(run_events))
        .route("/predict", post(predict).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)))
        .route("/predict/stream", get(predict_stream))
        .route("/val", post(val))
        .route("/uploads", post(create_upload))
        .route("/uploads/{id}", get(get_upload).delete(delete_upload))
        .route("/uploads/{id}/parts/{n}", put(upload_part).layer(DefaultBodyLimit::max(MAX_PART_BYTES)))
        .route("/uploads/{id}/complete", post(complete_upload))
        .route("/cloud/providers", get(cloud_providers))
        .route("/cloud/connections", get(cloud_connections).post(connect_cloud))
        .route("/cloud/connections/{id}", delete(disconnect_cloud))
        .route("/cloud/connections/{id}/check", post(check_cloud))
        .route("/cloud/connections/{id}/offers", get(cloud_offers))
        .route("/cloud/instances", get(cloud_instances))
        .route("/cloud/instances/{id}", delete(terminate_instance))
        .route("/label/agents", get(label_agents))
        .route("/label/jobs", get(label_jobs).post(create_label_job))
        .route("/label/jobs/{id}", get(label_job).put(update_label_job).delete(delete_label_job))
        .route(
            "/label/jobs/{id}/images",
            get(label_images).post(add_label_images).layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES)),
        )
        .route(
            "/label/jobs/{id}/images/{image_id}",
            get(label_image).put(update_label_image).delete(delete_label_image),
        )
        .route("/label/jobs/{id}/start", post(start_labeling))
        .route("/label/jobs/{id}/stop", post(stop_labeling))
        .route("/label/jobs/{id}/review", post(review_label_images))
        .route("/label/jobs/{id}/events", get(label_events))
        .route("/label/jobs/{id}/logs", get(label_logs))
        .route("/label/jobs/{id}/dataset", post(label_dataset))
        .merge(bench::routes())
        .fallback(api_not_found)
        .method_not_allowed_fallback(method_not_allowed)
}

/// Runs a store call on the blocking pool (stores may read files or render images).
pub(crate) async fn blocking<T, F>(st: &AppState, f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce(&dyn Store) -> Result<T> + Send + 'static,
{
    let store = st.store.clone();
    tokio::task::spawn_blocking(move || f(&*store))
        .await
        .map_err(|e| Error::internal(format!("task failed: {e}")))?
}

async fn api_not_found(req: Request) -> Error {
    Error::new(
        StatusCode::NOT_FOUND,
        "route_not_found",
        format!("no route for {} {}", req.method(), req.uri().path()),
    )
    .hint("see docs/API.md for the endpoints")
}

async fn method_not_allowed(req: Request) -> Error {
    Error::new(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        format!("{} is not allowed on {}", req.method(), req.uri().path()),
    )
}

#[derive(Deserialize)]
struct ThumbParams {
    thumb: Option<u32>,
}

/// Default number of lines of `GET /api/runs/{id}/logs`.
pub const DEFAULT_LOG_LINES: usize = 500;
/// Maximum number of lines of `GET /api/runs/{id}/logs`.
pub const MAX_LOG_LINES: usize = 5000;

#[derive(Deserialize)]
struct LogsParams {
    limit: Option<usize>,
    after: Option<u64>,
}
