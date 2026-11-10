//! Terminating instances, and the reaper: one thread that, at start and then every [`super::Tuning::reaper_every`],
//! looks at the instances of runs no process follows: it retries the terminations the provider has not confirmed,
//! terminates the instances that are still alive (also a `launching` record a crash left: looked up by its name,
//! dropped when the launch created nothing) and prunes old records. A process follows a run while it holds the run's
//! `cloud.lock` and releases the run's instance itself, so an instance whose launch is in flight (in this process or
//! another) is never touched.

use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::{CloudInstance, CloudInstanceStatus};

use super::{Core, files};
use crate::error::Result;

impl Core {
    /// Asks the provider once to terminate instance `id`: `terminated` when it confirms (also when the instance was
    /// gone already), else `terminating` with the error, for the reaper to retry. An instance without a provider id
    /// (its launch is in flight, or a crash interrupted it) is looked up by its name; when there is none, there is
    /// nothing to terminate.
    pub(crate) fn terminate_once(&self, id: &str) -> Result<CloudInstance> {
        let record = self.ledger.get(id)?;
        if !record.status.is_alive() {
            return Ok(record);
        }
        let record = self.ledger.update(id, |i| {
            if i.status.is_alive() {
                i.status = CloudInstanceStatus::Terminating;
            }
        })?;
        let terminated = self.connected(&record.connection_id).map_err(|e| e.body.message).and_then(|c| {
            let values = c.connection.values();
            let refs = match &record.provider_ref {
                Some(r) => vec![r.clone()],
                None => c.provider.find(&values, &record.name).map_err(|e| e.message)?,
            };
            refs.iter().try_for_each(|r| c.provider.terminate(&values, r)).map_err(|e| e.message)
        });
        match terminated {
            Ok(()) => {
                tracing::info!(instance = %id, "terminated the cloud instance");
                self.ledger.update(id, |i| {
                    i.status = CloudInstanceStatus::Terminated;
                    i.error = None;
                })
            }
            Err(message) => {
                tracing::warn!(instance = %id, "terminating the cloud instance failed (retried): {message}");
                self.ledger.update(id, |i| i.error = Some(message))
            }
        }
    }

    /// [`Core::terminate_once`] until the provider confirms or `wait` has passed; the record as it is then.
    pub(crate) fn terminate_within(&self, id: &str, wait: Duration) -> Result<CloudInstance> {
        let t0 = Instant::now();
        let mut pause = self.tuning.poll;
        loop {
            let record = self.terminate_once(id)?;
            if !record.status.is_alive() || t0.elapsed() >= wait {
                return Ok(record);
            }
            std::thread::sleep(pause.min(wait.saturating_sub(t0.elapsed())));
            pause = (pause * 2).min(self.tuning.poll_max);
        }
    }

    /// One pass of the reaper (see the module docs).
    pub(crate) fn sweep(&self) {
        for record in self.ledger.list() {
            if !record.status.is_alive() {
                continue;
            }
            // Followed: its run's process releases it. (A run folder that is gone is followed by nobody.)
            let run_dir = record.run_id.as_deref().map(|r| self.run_dir(r));
            match run_dir.as_deref().map(files::try_lock_run).transpose() {
                Ok(Some(None)) => continue,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(instance = %record.id, "checking who follows its run: {e}");
                    continue;
                }
                Ok(_) => {}
            }
            if record.status != CloudInstanceStatus::Terminating
                && record.provider_ref.is_none()
                && !self.launched(&record)
            {
                tracing::info!(instance = %record.id, "dropped the record of a launch that created nothing");
                self.ledger.remove(&record.id);
                continue;
            }
            if let Err(e) = self.terminate_once(&record.id) {
                tracing::warn!(instance = %record.id, "terminating it: {}", e.body.message);
            }
        }
    }

    /// Whether the interrupted launch of `record` created an instance (true when the provider cannot say, so the
    /// record stays and is looked at again).
    fn launched(&self, record: &CloudInstance) -> bool {
        let Ok(c) = self.connected(&record.connection_id) else { return true };
        !matches!(c.provider.find(&c.connection.values(), &record.name), Ok(refs) if refs.is_empty())
    }
}

/// Starts the reaper thread; it sweeps at once, then every `reaper_every`, until the process shuts down.
pub(crate) fn spawn(core: std::sync::Arc<Core>) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new().name("folo-cloud-reaper".into()).spawn(move || {
        loop {
            core.sweep();
            if core.halt.pause(core.tuning.reaper_every) {
                return;
            }
        }
    })
}
