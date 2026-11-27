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

/// Allowed top-level keys of [`TrainConfig`] in `POST /api/runs` and `POST /api/runs/estimate`.
const TRAIN_CONFIG_FIELDS: &[&str] = &[
    "name",
    "dataset_id",
    "arch",
    "epochs",
    "batch",
    "imgsz",
    "device",
    "optimizer",
    "lr0",
    "weights",
    "augment",
    "seed",
    "amp",
    "head",
    "compute",
];
/// Extra top-level control keys accepted by `POST /api/runs/estimate`.
const ESTIMATE_EXTRA_FIELDS: &[&str] = &["auto_batch", "auto_batch_target"];
/// Extra top-level control keys accepted by `POST /api/runs`.
const RUN_CONTROL_FIELDS: &[&str] =
    &["checkpoint", "resume_run_id", "auto_batch", "auto_batch_target", "max_preempt_retries"];
/// Allowed keys of [`folo_api::Augment`].
const AUGMENT_FIELDS: &[&str] = &["mosaic", "mixup", "hsv", "fliplr", "scale", "translate", "close_mosaic"];
/// Allowed keys of [`folo_api::Compute`].
const COMPUTE_FIELDS: &[&str] = &["connection_id", "offer_id", "max_hours", "disk_gb"];

/// Rejects keys in `obj` that are not in `allowed` (or `extra`), so misspelled training settings fail with `422
/// invalid_body` instead of silently falling back to defaults.
fn check_known_keys(
    prefix: &str,
    obj: &serde_json::Map<String, serde_json::Value>,
    allowed: &[&str],
    extra: &[&str],
) -> Result<()> {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) && !extra.contains(&key.as_str()) {
            let expected: Vec<&str> = allowed.iter().chain(extra.iter()).copied().collect();
            let qual = if prefix.is_empty() { key.clone() } else { format!("{prefix}.{key}") };
            return Err(body_error(format!(
                "unknown field `{qual}`, expected one of {}",
                expected.iter().map(|f| format!("`{f}`")).collect::<Vec<_>>().join(", ")
            )));
        }
    }
    Ok(())
}

/// The fields of a `POST /api/runs` or `POST /api/runs/estimate` body, with the check both make before its types.
fn run_fields(
    body: serde_json::Value,
    allow_run_controls: bool,
) -> Result<serde_json::Map<String, serde_json::Value>> {
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
    let extra = if allow_run_controls { RUN_CONTROL_FIELDS } else { ESTIMATE_EXTRA_FIELDS };
    check_known_keys("", &fields, TRAIN_CONFIG_FIELDS, extra)?;
    if let Some(serde_json::Value::Object(aug)) = fields.get("augment") {
        check_known_keys("augment", aug, AUGMENT_FIELDS, &[])?;
    }
    if let Some(serde_json::Value::Object(comp)) = fields.get("compute") {
        check_known_keys("compute", comp, COMPUTE_FIELDS, &[])?;
    }
    Ok(fields)
}

/// Extracts optional `batch: "auto"`, `auto_batch`, and `auto_batch_target` from a `POST /api/runs` body
/// before deserializing `folo_api::TrainConfig`.
fn extract_auto_batch(
    fields: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<crate::store::AutoBatchOptions> {
    let mut batch_auto = false;
    if let Some(val) = fields.get("batch")
        && let Some(s) = val.as_str()
    {
        if s.eq_ignore_ascii_case("auto") {
            batch_auto = true;
            fields.remove("batch");
        } else {
            return Err(body_error(format!(
                "batch: expected a positive integer in 1..=1024 or \"auto\", got {s:?}"
            )));
        }
    }
    let auto_flag = match fields.remove("auto_batch").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_bool().ok_or_else(|| body_error("auto_batch: expected a boolean"))?),
    };
    let target = match fields.remove("auto_batch_target").filter(|v| !v.is_null()) {
        None => None,
        Some(v) => Some(v.as_f64().ok_or_else(|| body_error("auto_batch_target: expected a number"))?),
    };
    if batch_auto && auto_flag == Some(false) {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            "`batch: \"auto\"` conflicts with `auto_batch: false`",
        ));
    }
    let opts = crate::store::AutoBatchOptions {
        enabled: batch_auto || auto_flag.unwrap_or(false),
        target_utilization: target,
    };
    crate::store::check_auto_batch_options(&opts)?;
    Ok(opts)
}

/// Extracts an optional `folo2-*` architecture string from `fields["arch"]`, normalizing the wire field to its
/// scale `folo-n..x` so `folo_api::TrainConfig` deserializes cleanly while the full `"folo2-*"` override is
/// carried alongside in [`crate::store::RunOptions`].
fn extract_arch_override(fields: &mut serde_json::Map<String, serde_json::Value>) -> Result<Option<String>> {
    let Some(val) = fields.get("arch") else { return Ok(None) };
    if val.is_null() {
        return Ok(None);
    }
    let Some(s) = val.as_str() else {
        return Err(body_error("arch: expected a string"));
    };
    let (api_arch, arch_override) = crate::store::parse_arch_override(s)?;
    if arch_override.is_some() {
        fields.insert("arch".into(), serde_json::json!(api_arch.as_str()));
    }
    Ok(arch_override)
}

/// Extracts an optional `max_preempt_retries` integer from a `POST /api/runs` body before deserializing
/// `folo_api::TrainConfig`.
fn extract_max_preempt_retries(
    fields: &mut serde_json::Map<String, serde_json::Value>,
) -> Result<Option<u32>> {
    match fields.remove("max_preempt_retries").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| body_error("max_preempt_retries: expected a non-negative integer")),
    }
}

/// `POST /api/runs`: a `TrainConfig` starts a new run (`201`); with `resume_run_id` it resumes that run instead
/// (`200`, same id); with `checkpoint` the new run continues that model's training state.
pub(super) async fn create_run(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Response> {
    let mut fields = run_fields(body, true)?;
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
    let auto_batch = extract_auto_batch(&mut fields)?;
    let arch_override = extract_arch_override(&mut fields)?;
    let max_preempt_retries = extract_max_preempt_retries(&mut fields)?;
    if checkpoint.is_some() && auto_batch.enabled {
        return Err(Error::new(
            StatusCode::BAD_REQUEST,
            "invalid_config",
            "auto_batch cannot be combined with `checkpoint` (a continued checkpoint keeps its batch and schedule)",
        ));
    }
    let config: TrainConfig =
        serde_json::from_value(serde_json::Value::Object(fields)).map_err(body_error)?;
    crate::store::check_max_preempt_retries(max_preempt_retries, config.compute.is_some())?;
    let options = crate::store::RunOptions { auto_batch, arch: arch_override, max_preempt_retries };
    let run = match checkpoint {
        Some(model) => blocking(&st, move |s| s.continue_run(&model, config)).await?,
        None if !options.is_default() => {
            blocking(&st, move |s| s.create_run_with_options(config, options)).await?
        }
        None => blocking(&st, move |s| s.create_run(config)).await?,
    };
    Ok((StatusCode::CREATED, Json(run)).into_response())
}

pub(super) async fn estimate_run(
    State(st): State<AppState>,
    ApiJson(body): ApiJson<serde_json::Value>,
) -> Result<Json<RunEstimate>> {
    let mut fields = run_fields(body, false)?;
    let auto_batch = extract_auto_batch(&mut fields)?;
    let arch_override = extract_arch_override(&mut fields)?;
    let config: TrainConfig =
        serde_json::from_value(serde_json::Value::Object(fields)).map_err(body_error)?;
    let options = crate::store::RunOptions { auto_batch, arch: arch_override, max_preempt_retries: None };
    Ok(Json(blocking(&st, move |s| s.estimate_run_with_options(config, options)).await?))
}

#[derive(serde::Serialize)]
struct RunWithPreemption {
    #[serde(flatten)]
    run: folo_api::Run,
    #[serde(skip_serializing_if = "Option::is_none")]
    cloud_preemption: Option<crate::store::CloudPreemptionStatus>,
}

pub(super) async fn get_run(State(st): State<AppState>, ApiPath(id): ApiPath<String>) -> Result<Response> {
    let (run, cloud_preemption) = blocking(&st, move |s| s.get_run_with_preemption(&id)).await?;
    Ok(Json(RunWithPreemption { run, cloud_preemption }).into_response())
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

    /// Keeps the strict unknown-field allowlists in sync with `folo_api::{TrainConfig, Augment, Compute}`.
    #[test]
    fn run_field_allowlists_match_api_structs() {
        let mut cfg: TrainConfig = serde_json::from_str(r#"{"dataset_id":"ds_1"}"#).unwrap();
        cfg.compute = Some(folo_api::Compute {
            connection_id: "cc_1".into(),
            offer_id: "off_1".into(),
            max_hours: 1.0,
            disk_gb: Some(50),
        });
        let val = serde_json::to_value(&cfg).unwrap();
        let top = val.as_object().unwrap();
        assert_eq!(top.len(), TRAIN_CONFIG_FIELDS.len(), "top-level TrainConfig fields drifted: {top:?}");
        for k in top.keys() {
            assert!(TRAIN_CONFIG_FIELDS.contains(&k.as_str()), "missing TrainConfig field `{k}`");
        }
        let aug = top["augment"].as_object().unwrap();
        assert_eq!(aug.len(), AUGMENT_FIELDS.len(), "Augment fields drifted: {aug:?}");
        for k in aug.keys() {
            assert!(AUGMENT_FIELDS.contains(&k.as_str()), "missing Augment field `{k}`");
        }
        let comp = top["compute"].as_object().unwrap();
        assert_eq!(comp.len(), COMPUTE_FIELDS.len(), "Compute fields drifted: {comp:?}");
        for k in comp.keys() {
            assert!(COMPUTE_FIELDS.contains(&k.as_str()), "missing Compute field `{k}`");
        }
    }

    #[test]
    fn extract_auto_batch_handles_batch_auto_flags_and_conflicts() {
        let mut obj = serde_json::json!({ "dataset_id": "ds_1", "batch": "auto", "auto_batch_target": 0.75 })
            .as_object()
            .unwrap()
            .clone();
        let ab = extract_auto_batch(&mut obj).unwrap();
        assert!(ab.enabled);
        assert_eq!(ab.target_utilization, Some(0.75));
        assert!(!obj.contains_key("batch"));
        assert!(!obj.contains_key("auto_batch_target"));

        // `batch: "auto"` + `auto_batch: false` conflicts.
        let mut conflict = serde_json::json!({ "dataset_id": "ds_1", "batch": "auto", "auto_batch": false })
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(extract_auto_batch(&mut conflict).unwrap_err().body.code, "invalid_config");

        // Non-auto string batch is rejected as invalid JSON body type.
        let mut bad_str =
            serde_json::json!({ "dataset_id": "ds_1", "batch": "huge" }).as_object().unwrap().clone();
        assert_eq!(extract_auto_batch(&mut bad_str).unwrap_err().body.code, "invalid_body");

        // `auto_batch_target` without auto-batch enabled is rejected.
        let mut no_ab = serde_json::json!({ "dataset_id": "ds_1", "auto_batch_target": 0.75 })
            .as_object()
            .unwrap()
            .clone();
        assert_eq!(extract_auto_batch(&mut no_ab).unwrap_err().body.code, "invalid_config");

        // Out-of-range `auto_batch_target` is rejected.
        let mut oob =
            serde_json::json!({ "dataset_id": "ds_1", "auto_batch": true, "auto_batch_target": 0.99 })
                .as_object()
                .unwrap()
                .clone();
        assert_eq!(extract_auto_batch(&mut oob).unwrap_err().body.code, "invalid_config");

        // `/api/runs/estimate` (`allow_run_controls = false`) accepts `auto_batch` and `auto_batch_target`
        // (and `batch: "auto"`) while still rejecting `resume_run_id`, `checkpoint`, and unknown typos.
        let mut est_ok = run_fields(
            serde_json::json!({
                "dataset_id": "ds_1",
                "batch": 24,
                "auto_batch": true,
                "auto_batch_target": 0.75
            }),
            false,
        )
        .unwrap();
        let est_ab = extract_auto_batch(&mut est_ok).unwrap();
        assert!(est_ab.enabled);
        assert_eq!(est_ab.target_utilization, Some(0.75));
        assert_eq!(est_ok.get("batch"), Some(&serde_json::json!(24)));

        let mut est_auto =
            run_fields(serde_json::json!({ "dataset_id": "ds_1", "batch": "auto" }), false).unwrap();
        assert!(extract_auto_batch(&mut est_auto).unwrap().enabled);

        let typo = run_fields(serde_json::json!({ "dataset_id": "ds_1", "auto_batch_targte": 0.75 }), false)
            .unwrap_err();
        assert_eq!(typo.status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(typo.body.code, "invalid_body");
    }
}
