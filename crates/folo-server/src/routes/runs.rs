//! `/api/runs`: creating, resuming, stopping and deleting runs, their history, logs and live events.

use std::collections::VecDeque;
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
use tokio::sync::broadcast::error::RecvError;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::wrappers::errors::BroadcastStreamRecvError;

use crate::AppState;
use crate::error::{ApiJson, ApiPath, ApiQuery, Error, Result};
use crate::store::{LogsQuery, MetricsQuery, QueueMove, RunSseStream, RunSseSubscription, SseFrame};
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

fn lag_warning_frame(dropped: u64) -> SseFrame {
    SseFrame::run(&RunEvent::Log(LogLine {
        t: rfc3339(unix_now()),
        level: LogLevel::Warn,
        message: format!(
            "client fell behind; {dropped} events dropped (reload /metrics for the full history)"
        ),
    }))
}

fn run_sse_frames(sub: RunSseSubscription) -> impl Stream<Item = SseFrame> {
    let rest: std::pin::Pin<Box<dyn Stream<Item = SseFrame> + Send>> = match sub.stream {
        RunSseStream::Frames(rx) => {
            let mut pending = VecDeque::with_capacity(2);
            pending.push_back(sub.initial);
            Box::pin(stream::unfold((pending, rx, sub.resync), |(mut pending, mut rx, resync)| async move {
                if let Some(frame) = pending.pop_front() {
                    return Some((frame, (pending, rx, resync)));
                }
                match rx.recv().await {
                    Ok(frame) => Some((frame, (pending, rx, resync))),
                    Err(RecvError::Closed) => None,
                    Err(RecvError::Lagged(dropped)) => {
                        let Some(resync_fn) = resync.as_ref() else {
                            return Some((lag_warning_frame(dropped), (pending, rx, resync)));
                        };
                        // Advance past stale buffered frames by atomically snapshotting and resubscribing at
                        // the live tail so a lagged client never observes older status transitions after the
                        // snapshot.
                        let skipped = dropped.saturating_add(rx.len() as u64);
                        let (status, new_rx) = resync_fn()?;
                        rx = new_rx;
                        pending.push_back(SseFrame::run(&RunEvent::Status(status)));
                        Some((lag_warning_frame(skipped), (pending, rx, resync)))
                    }
                }
            }))
        }
        RunSseStream::Events(rx) => {
            let first = stream::once(std::future::ready(sub.initial));
            let live = BroadcastStream::new(rx).map(|r| match r {
                Ok(ev) => SseFrame::run(&ev),
                Err(BroadcastStreamRecvError::Lagged(dropped)) => lag_warning_frame(dropped),
            });
            Box::pin(first.chain(live))
        }
    };
    rest
}

/// `GET /api/runs/{id}/events`: a `status` event, then live pre-serialized events until the client disconnects,
/// the run is deleted or the server shuts down. A subscriber that falls behind receives a warning `log` event
/// followed by a fresh `status` snapshot from the live tail so missed state transitions are recovered in order.
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::store::RunSseResync;
    use folo_api::{Loss, RunStatus, StatusEvent, Step};
    use tokio::sync::{Notify, broadcast};

    use super::*;

    #[tokio::test]
    async fn run_sse_lag_resyncs_to_live_tail_without_stale_status_transitions() {
        let (tx, rx1) = broadcast::channel::<SseFrame>(4);
        let rx2 = tx.subscribe();
        let initial = SseFrame::run(&RunEvent::Status(StatusEvent::new(RunStatus::Running, None)));
        let resynced = Arc::new(AtomicUsize::new(0));
        let ready = Arc::new(Notify::new());
        let tx_slot = Arc::new(std::sync::Mutex::new(Some(tx.clone())));
        let tx_resync = tx_slot.clone();
        let resynced_ref = resynced.clone();
        let ready_ref = ready.clone();
        let resync: RunSseResync = Arc::new(move || {
            let sub = tx_resync.lock().unwrap().as_ref()?.subscribe();
            if resynced_ref.fetch_add(1, Ordering::SeqCst) + 1 == 2 {
                ready_ref.notify_one();
            }
            Some((StatusEvent::new(RunStatus::Stopped, None), sub))
        });

        // Overflow the capacity-4 buffer so Step(1) is dropped while Stopping -> Log("final") -> Stopped stay
        // retained in the ring buffer. Resync must advance past those stale buffered transitions instead of
        // emitting `Stopped` followed by `Stopping`.
        let _ = tx.send(SseFrame::run(&RunEvent::Step(Step {
            step: 1,
            epoch: 1,
            epochs: 5,
            steps_per_epoch: 10,
            loss: Loss { box_: 1.0, cls: 1.0, dfl: 1.0, total: 3.0 },
            lr: 0.01,
            imgs_per_s: 10.0,
            eta_s: 5.0,
        })));
        let _ = tx.send(SseFrame::run(&RunEvent::Step(Step {
            step: 2,
            epoch: 1,
            epochs: 5,
            steps_per_epoch: 10,
            loss: Loss { box_: 0.9, cls: 0.9, dfl: 0.9, total: 2.7 },
            lr: 0.01,
            imgs_per_s: 10.0,
            eta_s: 4.0,
        })));
        let _ = tx.send(SseFrame::run(&RunEvent::Status(StatusEvent::new(RunStatus::Stopping, None))));
        let _ = tx.send(SseFrame::run(&RunEvent::Log(LogLine {
            t: "2026-01-01T00:00:00Z".into(),
            level: LogLevel::Info,
            message: "final".into(),
        })));
        let _ = tx.send(SseFrame::run(&RunEvent::Status(StatusEvent::new(RunStatus::Stopped, None))));

        let post = tokio::spawn({
            let tx_slot = tx_slot.clone();
            let ready = ready.clone();
            async move {
                ready.notified().await;
                if let Some(tx) = tx_slot.lock().unwrap().take() {
                    let _ = tx.send(SseFrame::run(&RunEvent::Log(LogLine {
                        t: "2026-01-01T00:00:01Z".into(),
                        level: LogLevel::Info,
                        message: "post-resync".into(),
                    })));
                }
            }
        });
        drop(tx);

        let (frames1, frames2): (Vec<SseFrame>, Vec<SseFrame>) = tokio::join!(
            run_sse_frames(RunSseSubscription {
                initial: initial.clone(),
                stream: RunSseStream::Frames(rx1),
                resync: Some(resync.clone()),
            })
            .collect(),
            run_sse_frames(RunSseSubscription {
                initial,
                stream: RunSseStream::Frames(rx2),
                resync: Some(resync),
            })
            .collect(),
        );
        post.await.unwrap();

        assert_eq!(
            frames1.iter().map(|f| f.event).collect::<Vec<_>>(),
            ["status", "log", "status", "log"],
            "stale buffered stopping/final/stopped frames must not follow the resync snapshot"
        );
        assert!(frames1[1].data.contains("5 events dropped"), "{}", frames1[1].data);
        let resync_status: StatusEvent = serde_json::from_str(&frames1[2].data).unwrap();
        assert_eq!(resync_status.status, RunStatus::Stopped);
        assert!(frames1[3].data.contains("post-resync"));
        // Concurrent subscribers share the exact same pre-serialized Arc<str> payload.
        assert!(Arc::ptr_eq(&frames1[3].data, &frames2[3].data));

        // When the run is deleted while lagged, resync returns None and the stream closes cleanly without
        // emitting stale buffered frames.
        let (tx2, rx_del) = broadcast::channel::<SseFrame>(1);
        let _ = tx2.send(SseFrame::new("log", "{}"));
        let _ = tx2.send(SseFrame::new("log", "{}"));
        drop(tx2);
        let del_frames: Vec<SseFrame> = run_sse_frames(RunSseSubscription {
            initial: SseFrame::run(&RunEvent::Status(StatusEvent::new(RunStatus::Running, None))),
            stream: RunSseStream::Frames(rx_del),
            resync: Some(Arc::new(|| None)),
        })
        .collect()
        .await;
        assert_eq!(del_frames.iter().map(|f| f.event).collect::<Vec<_>>(), ["status"]);
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
        let resync = sub1.resync.expect("real store provides lag resync callback");
        let (snap, mut rx3) = resync().expect("alive run resyncs");
        assert_eq!(snap.status, RunStatus::Queued);

        entry.log(LogLevel::Info, "shared payload");
        let f1 = rx1.recv().await.unwrap();
        let f2 = rx2.recv().await.unwrap();
        let f3 = rx3.recv().await.unwrap();
        assert_eq!(f1.event, "log");
        assert!(Arc::ptr_eq(&f1.data, &f2.data), "concurrent subscribers share one Arc<str>");
        assert!(Arc::ptr_eq(&f1.data, &f3.data), "resynced subscriber shares the same Arc<str>");

        // Dropping our local `Arc<RunEntry>` and deleting the run drops `sse_tx` so open streams close immediately.
        drop(entry);
        runs.delete(&id).unwrap();
        assert!(resync().is_none(), "weak resync handle returns None once deleted");
        assert!(matches!(rx1.recv().await, Err(broadcast::error::RecvError::Closed)));
        assert!(matches!(rx2.recv().await, Err(broadcast::error::RecvError::Closed)));
        assert!(matches!(rx3.recv().await, Err(broadcast::error::RecvError::Closed)));
        runs.shutdown(None);
    }
}
