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
use crate::store::{
    LogsQuery, MetricsQuery, QueueMove, RunSseStream, RunSseSubscription, RunStatusResync, SseFrame,
};
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

/// The fields of a `POST /api/runs` or `POST /api/runs/estimate` body, with the check both make before its types.
fn run_fields(body: serde_json::Value) -> Result<serde_json::Map<String, serde_json::Value>> {
    let serde_json::Value::Object(fields) = body else {
        return Err(body_error("expected a JSON object"));
    };
    // Distillation is a recipe key of the CLI (docs/MODEL.md §8.9), not part of the API's TrainConfig: refused, so a
    // body that asks for a teacher never trains (or is estimated) without one.
    if fields.contains_key("distill") {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            "distill (distillation from a teacher, docs/MODEL.md §8.9) is not a setting of the HTTP API",
        )
        .hint("train with a teacher from the CLI: folo train --teacher <weights> (docs/CLI.md)"));
    }
    Ok(fields)
}

/// `POST /api/runs`: a `TrainConfig` starts a new run (`201`); with `resume_run_id` it resumes that run instead
/// (`200`, same id); with `checkpoint` the new run continues that model's training state.
pub(super) async fn create_run(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Response> {
    let mut fields = run_fields(body)?;
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
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Json<RunEstimate>> {
    let fields = run_fields(body)?;
    let config: TrainConfig =
        serde_json::from_value(serde_json::Value::Object(fields)).map_err(body_error)?;
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

fn lagged_run_frames(dropped: u64, current: Option<&RunStatusResync>) -> Vec<SseFrame> {
    let mut out = Vec::with_capacity(2);
    out.push(SseFrame::run(&RunEvent::Log(LogLine {
        t: rfc3339(unix_now()),
        level: LogLevel::Warn,
        message: format!(
            "client fell behind; {dropped} events dropped (reload /metrics for the full history)"
        ),
    })));
    if let Some(status) = current.and_then(|f| f()) {
        out.push(SseFrame::run(&RunEvent::Status(status)));
    }
    out
}

fn run_sse_frames(sub: RunSseSubscription) -> impl Stream<Item = SseFrame> {
    let first = stream::once(std::future::ready(sub.initial));
    let current = sub.current;
    let rest: std::pin::Pin<Box<dyn Stream<Item = SseFrame> + Send>> = match sub.stream {
        RunSseStream::Frames(rx) => Box::pin(BroadcastStream::new(rx).flat_map(move |r| {
            let frames = match r {
                Ok(frame) => vec![frame],
                Err(BroadcastStreamRecvError::Lagged(n)) => lagged_run_frames(n, current.as_ref()),
            };
            stream::iter(frames)
        })),
        RunSseStream::Events(rx) => Box::pin(BroadcastStream::new(rx).flat_map(move |r| {
            let frames = match r {
                Ok(ev) => vec![SseFrame::run(&ev)],
                Err(BroadcastStreamRecvError::Lagged(n)) => lagged_run_frames(n, current.as_ref()),
            };
            stream::iter(frames)
        })),
    };
    first.chain(rest)
}

/// `GET /api/runs/{id}/events`: a `status` event, then live pre-serialized events until the client disconnects,
/// the run is deleted or the server shuts down. A subscriber that falls behind receives a warning `log` event
/// followed by a fresh `status` snapshot so missed state transitions are recovered cleanly.
pub(super) async fn run_events(
    State(st): State<AppState>,
    ApiPath(id): ApiPath<String>,
) -> Result<Sse<impl Stream<Item = std::result::Result<Event, Infallible>>>> {
    let sub = st.store.subscribe_sse(&id)?;
    let mut shutdown = st.shutdown.subscribe();
    let stopped = async move {
        let _ = shutdown.wait_for(|v| *v).await;
    };
    let events = run_sse_frames(sub).map(|frame| Ok(frame.to_sse())).take_until(stopped);
    Ok(Sse::new(events).keep_alive(KeepAlive::new().interval(KEEP_ALIVE).text("keep-alive")))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use folo_api::{RunStatus, StatusEvent};
    use tokio::sync::broadcast;

    use super::*;

    #[tokio::test]
    async fn run_sse_lag_emits_warning_and_current_status_then_resumes() {
        let (tx, rx1) = broadcast::channel::<SseFrame>(2);
        let rx2 = tx.subscribe();
        let initial = SseFrame::run(&RunEvent::Status(StatusEvent::new(RunStatus::Running, None)));
        let alive = Arc::new(AtomicBool::new(true));
        let alive_ref = alive.clone();
        let current: RunStatusResync = Arc::new(move || {
            alive_ref.load(Ordering::SeqCst).then(|| StatusEvent::new(RunStatus::Completed, None))
        });

        // Overflow the capacity-2 buffer with 4 frames so both subscribers lag by 2.
        for i in 0..4 {
            let frame = SseFrame::run(&RunEvent::Log(LogLine {
                t: "2026-01-01T00:00:00Z".into(),
                level: LogLevel::Info,
                message: format!("msg-{i}"),
            }));
            let _ = tx.send(frame);
        }
        drop(tx);

        let frames1: Vec<SseFrame> = run_sse_frames(RunSseSubscription {
            initial: initial.clone(),
            stream: RunSseStream::Frames(rx1),
            current: Some(current.clone()),
        })
        .collect()
        .await;
        let frames2: Vec<SseFrame> = run_sse_frames(RunSseSubscription {
            initial,
            stream: RunSseStream::Frames(rx2),
            current: Some(current),
        })
        .collect()
        .await;

        assert_eq!(frames1[0].event, "status");
        assert_eq!(frames1[1].event, "log");
        assert!(frames1[1].data.contains("2 events dropped"), "{}", frames1[1].data);
        assert_eq!(frames1[2].event, "status");
        let resync: StatusEvent = serde_json::from_str(&frames1[2].data).unwrap();
        assert_eq!(resync.status, RunStatus::Completed);
        assert_eq!(frames1[3].event, "log");
        assert!(frames1[3].data.contains("msg-2"));
        assert_eq!(frames1[4].event, "log");
        assert!(frames1[4].data.contains("msg-3"));
        // Concurrent subscribers share the exact same pre-serialized Arc<str> payload.
        assert!(Arc::ptr_eq(&frames1[3].data, &frames2[3].data));
        assert!(Arc::ptr_eq(&frames1[4].data, &frames2[4].data));

        // When the run is deleted while lagged, resync omits status and stream terminates cleanly.
        alive.store(false, Ordering::SeqCst);
        let (tx2, rx_del) = broadcast::channel::<SseFrame>(1);
        let _ = tx2.send(SseFrame::new("log", "{}"));
        let _ = tx2.send(SseFrame::new("log", "{}"));
        drop(tx2);
        let del_frames: Vec<SseFrame> = run_sse_frames(RunSseSubscription {
            initial: SseFrame::run(&RunEvent::Status(StatusEvent::new(RunStatus::Running, None))),
            stream: RunSseStream::Frames(rx_del),
            current: Some(Arc::new(move || None)),
        })
        .collect()
        .await;
        assert_eq!(del_frames.iter().map(|f| f.event).collect::<Vec<_>>(), ["status", "log", "log"]);
    }

    #[tokio::test]
    async fn real_runs_subscribe_sse_shares_frames_and_closes_on_delete() {
        let dir = folo_test_support::TestDir::new("runs-sse-shared");
        let runs = crate::runs::Runs::open(
            dir.to_path_buf(),
            Arc::new(crate::throughput::Throughput::in_memory(Default::default())),
        );
        let config: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1","name":"s"}"#).unwrap();
        let dataset = crate::runs::DatasetRef {
            id: "ds_1".into(),
            name: "d".into(),
            path: "nowhere".into(),
            format: folo_api::DatasetFormat::Yolo,
            classes: vec![],
        };
        let entry = runs.register(config, dataset, None, None, 0, "cpu".into(), None).unwrap();
        let id = entry.id().to_string();

        let sub1 = runs.subscribe_sse(&id).unwrap();
        let sub2 = runs.subscribe_sse(&id).unwrap();
        let RunSseStream::Frames(mut rx1) = sub1.stream else { panic!("expected Frames stream") };
        let RunSseStream::Frames(mut rx2) = sub2.stream else { panic!("expected Frames stream") };
        let current = sub1.current.expect("real store provides lag resync callback");
        assert!(current().is_some());

        entry.log(LogLevel::Info, "shared payload");
        let f1 = rx1.recv().await.unwrap();
        let f2 = rx2.recv().await.unwrap();
        assert_eq!(f1.event, "log");
        assert!(Arc::ptr_eq(&f1.data, &f2.data), "concurrent subscribers share one Arc<str>");

        // Dropping our local `Arc<RunEntry>` and deleting the run drops `sse_tx` so open streams close immediately.
        drop(entry);
        runs.delete(&id).unwrap();
        assert!(current().is_none(), "weak resync handle returns None once deleted");
        assert!(matches!(rx1.recv().await, Err(broadcast::error::RecvError::Closed)));
        assert!(matches!(rx2.recv().await, Err(broadcast::error::RecvError::Closed)));
        runs.shutdown(None);
    }
}
