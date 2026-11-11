//! `/api/runs`: creating, resuming, stopping and deleting runs, their history, logs and live events.

use std::convert::Infallible;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use folo_api::{
    CliCommand, LogLevel, LogLine, QueueRequest, RunEstimate, RunEvent, RunList, RunLogs, TrainConfig,
};
use futures_util::{Stream, StreamExt, stream};
use serde::Deserialize;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{LogsQuery, MetricsQuery, QueueMove};
use crate::util::{rfc3339, unix_now};

use super::{DEFAULT_LOG_LINES, KEEP_ALIVE, LogsParams, MAX_LOG_LINES, blocking};

// ---- runs --------------------------------------------------------------------------------------------------------

pub(super) async fn list_runs(State(st): State<AppState>) -> Result<Json<RunList>> {
    Ok(Json(RunList { runs: st.store.list_runs()? }))
}

fn body_error(e: impl std::fmt::Display) -> Error {
    Error::new(
        StatusCode::UNPROCESSABLE_ENTITY,
        "invalid_body",
        format!("Failed to deserialize the JSON body into the target type: {e}"),
    )
}

/// A string field of a `POST /api/runs` body (`null` counts as missing).
fn string_field(fields: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<Option<String>> {
    match fields.get(key).filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => {
            Ok(Some(v.as_str().ok_or_else(|| body_error(format!("{key}: expected a string")))?.into()))
        }
    }
}

/// `POST /api/runs`: a `TrainConfig` starts a new run (`201`); with `resume_run_id` it resumes that run instead
/// (`200`, same id); with `checkpoint` the new run continues that model's training state.
pub(super) async fn create_run(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Response> {
    let serde_json::Value::Object(mut fields) = body else {
        return Err(body_error("expected a JSON object"));
    };
    let checkpoint = string_field(&fields, "checkpoint")?;
    let resume = string_field(&fields, "resume_run_id")?;
    if checkpoint.is_some() {
        let with = if resume.is_some() {
            Some("resume_run_id")
        } else {
            string_field(&fields, "weights")?.map(|_| "weights")
        };
        if let Some(with) = with {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                format!("checkpoint cannot be combined with {with}"),
            )
            .hint(
                "checkpoint starts a new run from a checkpoint's training state; leave the other field out",
            ));
        }
    }
    fields.remove("checkpoint");
    if let Some(rid) = resume {
        let run = blocking(&st, move |s| s.resume_run(&rid, &fields)).await?;
        return Ok(Json(run).into_response());
    }
    let config: TrainConfig =
        serde_json::from_value(serde_json::Value::Object(fields)).map_err(body_error)?;
    let run = match checkpoint {
        Some(model) => blocking(&st, move |s| s.continue_run(&model, config)).await?,
        None => blocking(&st, move |s| s.create_run(config)).await?,
    };
    Ok((StatusCode::CREATED, Json(run)).into_response())
}

pub(super) async fn estimate_run(
    State(st): State<AppState>,
    ApiJson(config): ApiJson<TrainConfig>,
) -> Result<Json<RunEstimate>> {
    Ok(Json(blocking(&st, move |s| s.estimate_run(config)).await?))
}

pub(super) async fn get_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(st.store.get_run(&id)?).into_response())
}

pub(super) async fn run_cli(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Json<CliCommand>> {
    Ok(Json(blocking(&st, move |s| s.run_cli(&id)).await?))
}

pub(super) async fn stop_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.stop_run(&id)).await?).into_response())
}

/// `POST /api/runs/{id}/queue`: moves a waiting run in the training queue (or starts it next).
pub(super) async fn move_run(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiJson(req): ApiJson<QueueRequest>,
) -> Result<Response> {
    let mv = QueueMove::parse(req)?;
    Ok(Json(blocking(&st, move |s| s.move_run(&id, mv)).await?).into_response())
}

pub(super) async fn delete_run(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<StatusCode> {
    blocking(&st, move |s| s.delete_run(&id)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[derive(Deserialize)]
pub(super) struct MetricsParams {
    every: Option<u64>,
    after_step: Option<u64>,
    after_epoch: Option<u32>,
}

pub(super) async fn run_metrics(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<MetricsParams>,
) -> Result<Response> {
    let q = MetricsQuery {
        every: p.every.unwrap_or(1).max(1),
        after_step: p.after_step,
        after_epoch: p.after_epoch,
    };
    Ok(Json(blocking(&st, move |s| s.run_metrics(&id, &q)).await?).into_response())
}

#[derive(Deserialize)]
pub(super) struct SamplesParams {
    epoch: Option<u32>,
}

pub(super) async fn run_samples(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<SamplesParams>,
) -> Result<Response> {
    Ok(Json(blocking(&st, move |s| s.run_samples(&id, p.epoch)).await?).into_response())
}

pub(super) async fn run_logs(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
    ApiQuery(p): ApiQuery<LogsParams>,
) -> Result<Json<RunLogs>> {
    let q = LogsQuery { limit: p.limit.unwrap_or(DEFAULT_LOG_LINES).clamp(1, MAX_LOG_LINES), after: p.after };
    Ok(Json(blocking(&st, move |s| s.run_logs(&id, &q)).await?))
}

fn sse_event(ev: &RunEvent) -> Event {
    Event::default().event(ev.name()).data(ev.data_json())
}

/// `GET /api/runs/{id}/events`: a `status` event, then live events until the client disconnects, the run is
/// deleted or the server shuts down. Each client owns only a broadcast receiver; dropping the response stream drops
/// it, and the run's event pump stops by itself once no receivers are left.
pub(super) async fn run_events(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let sub = st.store.subscribe(&id)?;
    let mut shutdown = st.shutdown.subscribe();
    let first = stream::once(std::future::ready(RunEvent::Status(sub.initial)));
    let rest = BroadcastStream::new(sub.events).map(|r| match r {
        Ok(ev) => ev,
        Err(BroadcastStreamRecvError::Lagged(n)) => RunEvent::Log(LogLine {
            t: rfc3339(unix_now()),
            level: LogLevel::Warn,
            message: format!("client fell behind; {n} events dropped (reload /metrics for the full history)"),
        }),
    });
    let stopped = async move {
        let _ = shutdown.wait_for(|v| *v).await;
    };
    let events = first.chain(rest).map(|ev| Ok(sse_event(&ev))).take_until(stopped);
    Ok(Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive")))
}
