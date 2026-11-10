//! Labeling runs: starting one on its own thread, the agent it asks, and stopping it.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::http::StatusCode;
use folo_api::{
    LabelAgentId, LabelImageStatus, LabelJob, LabelRun, LabelScope, LabelUsage, LogLevel, StartLabeling,
};

use crate::error::{Error, Result};
use crate::util::{rfc3339, unix_now};

use super::agent::Agent;
use super::job::Active;
use super::validate::{check_model, check_text};
use super::{DEFAULT_PARALLEL, Labeler, MAX_BATCH, MAX_NOTE_CHARS, MAX_PARALLEL, invalid, run};

impl Labeler {
    /// `POST /api/label/jobs/{id}/start`: starts an agent run on its own thread.
    pub fn start(&self, id: &str, req: StartLabeling) -> Result<LabelJob> {
        let note = check_text("note", &req.note, MAX_NOTE_CHARS)?;
        let model = check_model(req.model.as_deref())?;
        let model = model.as_deref();
        if req.batch.is_some_and(|b| !(1..=MAX_BATCH).contains(&b)) {
            return Err(invalid(format!("batch must be in 1..={MAX_BATCH}")));
        }
        if req.parallel.is_some_and(|p| !(1..=MAX_PARALLEL).contains(&p)) {
            return Err(invalid(format!("parallel must be in 1..={MAX_PARALLEL}")));
        }
        let job = self.job(id)?;
        let agent = self.agent(req.agent)?;
        let info = agent.info();
        let model = model.map(str::to_string).or(info.default_model);
        let effort = req.effort.unwrap_or(info.default_effort);

        // Looked up before the agent was probed, which can take a while: the job may be gone by now.
        let mut state = job.live()?;
        state.check_idle("start another run")?;
        let ids = match &req.image_ids {
            Some(ids) => {
                let mut seen = HashSet::new();
                for id in ids {
                    state
                        .record(id)
                        .map_err(|_| invalid(format!("image_ids: the job has no image '{id}'")))?;
                }
                ids.iter().filter(|id| seen.insert(id.as_str())).cloned().collect()
            }
            None => state
                .file
                .images
                .iter()
                .filter(|r| req.scope == LabelScope::All || r.status != LabelImageStatus::Labeled)
                .map(|r| r.id.clone())
                .collect::<Vec<_>>(),
        };
        if ids.is_empty() {
            let why =
                if state.file.images.is_empty() { "the job has no images" } else { "every image is labeled" };
            return Err(invalid(format!("nothing to label: {why}"))
                .hint("add images, or send \"scope\": \"all\" to label every image again"));
        }
        let batch = req.batch.map_or(agent.default_batch(), |b| b as usize);
        let plan = run::Plan {
            agent: agent.clone(),
            model: model.clone(),
            effort,
            note,
            batches: ids.chunks(batch).map(<[String]>::to_vec).collect(),
            parallel: req.parallel.unwrap_or(DEFAULT_PARALLEL) as usize,
            cancel: Arc::new(AtomicBool::new(false)),
            save_every: self.save_every,
        };
        let previous = state.file.run.replace(LabelRun {
            agent: req.agent,
            model: model.clone(),
            effort,
            started: rfc3339(unix_now()),
            finished: None,
            total: ids.len() as u64,
            done: 0,
            failed: 0,
            outcome: None,
            error: None,
            hint: None,
            usage: LabelUsage::default(),
        });
        let calls = plan.batches.len();
        let message = format!(
            "{} ({}, {} effort) starts on {} {}: {calls} {} of up to {batch}",
            req.agent.display_name(),
            model.as_deref().unwrap_or("its default model"),
            effort.as_str(),
            ids.len(),
            if ids.len() == 1 { "image" } else { "images" },
            if calls == 1 { "call" } else { "calls" },
        );
        state.active = Some(Active {
            stopping: false,
            cancel: plan.cancel.clone(),
            waiting: ids.into_iter().collect(),
            in_call: HashSet::new(),
            thread: None,
        });
        // The thread's first step takes the job's lock, so it starts working once this call has let go of it.
        let worker = job.clone();
        let started = job.save(&state).and_then(|()| {
            std::thread::Builder::new()
                .name(format!("folo-label-{id}"))
                .spawn(move || run::run(&worker, &plan))
                .map_err(|e| Error::internal(format!("could not start the run's thread: {e}")))
        });
        match started {
            Ok(thread) => state.active.as_mut().expect("set above").thread = Some(thread),
            Err(e) => {
                // Nothing runs: the job is as it was.
                state.active = None;
                state.file.run = previous;
                let _ = job.save(&state);
                return Err(e);
            }
        }
        job.log(&mut state, LogLevel::Info, message);
        job.emit_job(&state);
        Ok(state.job())
    }

    /// The agent `id`, installed: `501 not_supported` with how to get it otherwise. Vision benchmarks ask the
    /// same agents ([`crate::bench`]), so scripted agents script them too.
    pub fn agent(&self, id: LabelAgentId) -> Result<Arc<dyn Agent>> {
        let unsupported = |message: String| Error::new(StatusCode::NOT_IMPLEMENTED, "not_supported", message);
        let Some(agent) = self.agents.iter().find(|a| a.id() == id) else {
            return Err(unsupported(format!("this server has no {} agent", id.display_name())));
        };
        let info = agent.info();
        if !info.available {
            let e = unsupported(format!("{} is not installed on the server's machine", info.name));
            return Err(match info.hint {
                Some(hint) => e.hint(hint),
                None => e,
            });
        }
        Ok(agent.clone())
    }

    /// `POST /api/label/jobs/{id}/stop`: ends the run's calls; the run's thread closes the run.
    pub fn stop(&self, id: &str) -> Result<LabelJob> {
        let job = self.job(id)?;
        let mut state = job.live()?;
        let Some(active) = state.active.as_mut() else {
            return Err(Error::conflict(
                "label_job_not_active",
                format!("labeling job '{id}' has no agent run"),
            )
            .hint("nothing to stop; GET /api/label/jobs/{id} has the last run's result"));
        };
        if !active.stopping {
            active.stopping = true;
            active.cancel.store(true, Ordering::Relaxed);
            job.log(&mut state, LogLevel::Info, "stop requested");
            job.emit_job(&state);
        }
        Ok(state.job())
    }
}
