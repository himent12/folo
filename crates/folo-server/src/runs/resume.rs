//! Resuming a stopped or failed run, here or on a cloud connection.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use axum::http::StatusCode;
use folo_api::{LogLevel, Run, RunCloud, RunEvent, RunStatus, TrainConfig};

use crate::engine;
use crate::error::{Error, Result};

use crate::util::{parse_rfc3339, unix_now};

use super::entry::status_event;
use super::{RunEntry, RunState, Runs, add_paused};

impl Runs {
    /// What `POST /api/runs` with `resume_run_id` asks of a `stopped` or `failed` run: the run, and its config with
    /// the request's `fields` (the body's other fields), which must each equal the run's config (an empty `name`
    /// counts as omitted), except `device` and `compute`, which may change. Nothing changes yet: [`Runs::resume`]
    /// (on this machine) or [`Runs::resume_cloud`] does.
    pub fn resume_config(
        &self,
        id: &str,
        fields: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(Arc<RunEntry>, TrainConfig)> {
        self.check_open()?;
        let e = self.get(id)?;
        let st = e.state.lock().unwrap();
        resumable(&st, id)?;
        let stored = st.file.run.config.clone();
        let mut merged = serde_json::to_value(&stored).expect("configs serialize");
        for (k, v) in fields {
            if k == "resume_run_id" || (k == "name" && v.as_str() == Some("")) {
                continue;
            }
            merged[k.as_str()] = v.clone();
        }
        let wanted: TrainConfig = serde_json::from_value(merged)
            .map_err(|err| Error::new(StatusCode::UNPROCESSABLE_ENTITY, "invalid_body", err.to_string()))?;
        let (a, b) = (serde_json::to_value(&stored).unwrap(), serde_json::to_value(&wanted).unwrap());
        let differ: Vec<&str> = a
            .as_object()
            .unwrap()
            .iter()
            .filter(|(k, v)| !matches!(k.as_str(), "device" | "compute") && b.get(k.as_str()) != Some(v))
            .map(|(k, _)| k.as_str())
            .collect();
        if !differ.is_empty() {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_config",
                format!("resuming keeps the run's config; {} differ(s) from it", differ.join(", ")),
            )
            .hint("omit the fields, or start a new run (with `weights` to start from this run's model)"));
        }
        if !Path::new(&st.file.dataset.path).exists() {
            return Err(Error::not_found("dataset", &st.file.dataset.id)
                .hint(format!("the run's dataset is gone from {}", st.file.dataset.path)));
        }
        drop(st);
        Ok((e, wanted))
    }

    /// Resumes run `e` on this machine with `wanted` ([`Runs::resume_config`]): from `weights/last.folo` when it
    /// exists (the full checkpoint state, same id), from scratch otherwise.
    pub fn resume(self: &Arc<Self>, e: &Arc<RunEntry>, wanted: TrainConfig) -> Result<Run> {
        let device = engine::resolve(&wanted.device)?;
        // A mixed-precision run resumes in mixed precision (`amp` must match): refused before it is queued where this
        // server cannot train it, as a new run is.
        if wanted.amp
            && let Err(err) = folo_vision::train::amp_available(device)
        {
            let mut err = engine::engine_error(err);
            err.body.hint = Some(
                "the run trained in mixed precision and resumes in it: resume it on a server with tensor cores \
                 (`folo serve --tensor-cores` / `folo mcp --tensor-cores` on a GPU that has them), or start a new \
                 f32 run with `weights` set to its model"
                    .into(),
            );
            return Err(err);
        }
        self.restart(e, wanted, None)?;
        self.enqueue(e)?;
        Ok(self.record(e))
    }

    /// Resumes run `e` on a cloud connection with `wanted` ([`Runs::resume_config`]; `compute` set): queued again
    /// with a fresh `cloud`, and claimed for [`crate::cloud`] to execute.
    pub(crate) fn resume_cloud(
        &self,
        e: &Arc<RunEntry>,
        wanted: TrainConfig,
        cloud: RunCloud,
    ) -> Result<Run> {
        self.restart(e, wanted, Some(cloud))
    }

    /// Queues run `e` again with `wanted`'s `device` and `compute` (and `cloud`, for a cloud run), unless it became
    /// active since [`Runs::resume_config`] looked at it.
    pub(super) fn restart(
        &self,
        e: &Arc<RunEntry>,
        wanted: TrainConfig,
        cloud: Option<RunCloud>,
    ) -> Result<Run> {
        let mut st = e.state.lock().unwrap();
        resumable(&st, &e.id)?;
        let last = e.dir.join("weights").join("last.folo");
        st.claimed = cloud.is_some();
        // Not training from its end (or from a resume it has not trained since: it was stopped while it waited) until
        // it trains again: the first part now, the wait in the queue once it starts (`RunEntry::on_event`).
        let at = unix_now().floor();
        let file = &mut st.file;
        if file.run.started.is_some() {
            let since = file.resumed_at.or_else(|| file.run.finished.as_deref().and_then(parse_rfc3339));
            if let Some(since) = since {
                add_paused(&mut file.run, at - since);
            }
            file.resumed_at = Some(at);
        }
        let r = &mut st.file.run;
        r.config.device = wanted.device;
        r.config.compute = wanted.compute;
        // `cloud` is there exactly for a cloud run: one resumed here is a run of this machine from now on (what its
        // instances cost is in its log and in the ledger).
        r.cloud = cloud;
        r.status = RunStatus::Queued;
        r.finished = None;
        r.error = None;
        r.error_code = None;
        r.eta_s = None;
        st.tail.clear();
        st.attempt += 1;
        e.stop.store(false, Ordering::Relaxed);
        e.persist(&st);
        e.send(RunEvent::Status(status_event(&st.file.run)));
        let run = st.file.shown();
        drop(st);
        e.log(
            LogLevel::Info,
            if last.is_file() {
                "resuming from weights/last.folo".to_string()
            } else {
                "no epoch finished before; starting over".to_string()
            },
        );
        Ok(run)
    }
}

/// Refuses to resume a run that is deleted, still active (or claimed) or completed.
fn resumable(st: &RunState, id: &str) -> Result<()> {
    if st.deleted {
        return Err(Error::not_found("run", id));
    }
    let status = st.file.run.status;
    if status.is_active() || st.claimed {
        return Err(Error::conflict("run_active", format!("run '{id}' is still active")));
    }
    if status == RunStatus::Completed {
        return Err(Error::conflict("run_not_resumable", format!("run '{id}' already completed"))
            .hint("start a new run with `weights` set to its model to train further"));
    }
    Ok(())
}
