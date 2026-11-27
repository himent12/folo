//! Training: following the worker's run until it ends, the instance's state and cost, and stopping it.

use std::time::Instant;

use folo_api::{CloudInstanceStatus, LogLevel, Run, RunStatus};
use folo_client::{Client, encode};
use folo_cloud::InstanceState;

use crate::cloud::state::Outcome;
use crate::util::{rfc3339, unix_now};

use super::{ATTEMPTS, COST_EVERY, Exec, Halt, failed};

impl Exec {
    /// Mirrors the worker's run until it ends (stopping it when asked to or at the time limit), then collects.
    pub(super) fn train(&mut self, client: &Client) -> std::result::Result<Outcome, Halt> {
        let run_id = self.private.run_id.clone().expect("a training cloud run has a worker run");
        let t = self.core.tuning.clone();
        let mut stop_requested_at: Option<Instant> = None;
        let mut stop_acked = false;
        let mut stop_warned = false;
        let mut time_up = false;
        let mut wait = t.poll;
        let (mut unreachable, mut gone): (Option<Instant>, Option<Instant>) = (None, None);
        let mut answered = false;
        let mut cost_at = Instant::now();
        let end = loop {
            // A copy that finished is taken whatever this poll finds: a worker lost right after still leaves it.
            self.reap_copy();
            if self.core.halt.is_on() {
                return Err(Halt::LetGo);
            }
            if self.instance_ended() {
                return Err(self.lost_by_hand());
            }
            let stopping = self.entry.stop_requested() || self.entry.run().status == RunStatus::Stopping;
            if stop_requested_at.is_none() && (stopping || self.past_deadline()) {
                time_up = !stopping;
                if time_up {
                    self.entry.log(
                        LogLevel::Warn,
                        format!("the time limit of {} is up; stopping the worker's run", self.hours()),
                    );
                }
                stop_requested_at = Some(Instant::now());
            }
            if let Some(requested_at) = stop_requested_at {
                let stop_deadline = requested_at + t.lost_grace;
                if Instant::now() >= stop_deadline {
                    // A worker that has not answered for most of that time has nothing to collect from: the run is
                    // lost with its instance (a time limit that ended while nothing followed the run, and the
                    // instance with it). Half the grace, not all of it: the silence is noticed a poll after the
                    // stop was sent.
                    if unreachable.is_some_and(|u| u.elapsed() >= t.lost_grace / 2) {
                        return Err(failed(
                            "cloud_instance_lost",
                            "the worker stopped answering before the run could be stopped and collected; the \
                             weights copied so far stay (resume the run to continue from them)",
                        ));
                    }
                    // Check the worker's terminal status one final time so a run that completed or failed right
                    // as the grace window closed is not misreported as `Stopped`.
                    if let Ok(Some(end)) = self.mirror(client, &run_id) {
                        if self.entry.run().cloud.is_some_and(|c| c.unanswered_since.is_some()) {
                            self.entry.update_cloud(|c| c.unanswered_since = None);
                        }
                        break Some(end);
                    }
                    self.entry
                        .log(LogLevel::Warn, "the worker's run did not stop in time; collecting what it has");
                    break None;
                }
                // Retry delivering `/stop` until the worker acknowledges it (or replies `run_not_active`),
                // bounded by `stop_deadline` so a transient outage never leaves the GPU training until
                // `lost_grace` expires nor delays checking the worker's terminal status past the deadline.
                if !stop_acked {
                    let attempts = if stop_warned { 1 } else { ATTEMPTS };
                    stop_acked =
                        self.stop_worker_run(client, &run_id, attempts, stop_deadline, !stop_warned)?;
                    if !stop_acked {
                        stop_warned = true;
                    }
                }
            }
            let mirrored = self.mirror(client, &run_id);
            // The worker answered: no silence is said any more, also one a process before this one recorded (the run
            // persisted with it) and on the answer that the run ended (the collection may take minutes).
            if mirrored.is_ok() && self.entry.run().cloud.is_some_and(|c| c.unanswered_since.is_some()) {
                self.entry.update_cloud(|c| c.unanswered_since = None);
            }
            match mirrored {
                Ok(Some(end)) => break Some(end),
                Ok(None) => {
                    if unreachable.is_some() {
                        self.entry.log(LogLevel::Info, "the worker answers again");
                    }
                    // A run followed again whose record does not say when its worker first answered (a server
                    // of before wrote it, or that write failed): it answers now. The first time stays.
                    if !std::mem::replace(&mut answered, true)
                        && let Some(id) = &self.private.instance_id
                    {
                        let _ = self.core.ledger.booted(id);
                    }
                    (wait, unreachable, gone) = (t.poll, None, None);
                    // If the worker just came back and `/stop` was never delivered, send it immediately within the
                    // remaining grace window.
                    if let Some(requested_at) = stop_requested_at
                        && !stop_acked
                    {
                        let stop_deadline = requested_at + t.lost_grace;
                        if Instant::now() < stop_deadline {
                            stop_acked =
                                self.stop_worker_run(client, &run_id, 1, stop_deadline, !stop_warned)?;
                            if !stop_acked {
                                stop_warned = true;
                            }
                        }
                    }
                }
                Err(e) if e.code() == "run_not_found" => {
                    if self.instance_ended() {
                        return Err(self.lost_by_hand());
                    }
                    let msg = "the worker no longer has the run (its server restarted or was replaced)";
                    if self.c.provider.info().provisions
                        && stop_requested_at.is_none()
                        && !self.is_stopping()
                        && !self.past_deadline()
                    {
                        return Err(Halt::Preempted(msg.into()));
                    }
                    return Err(failed("cloud_instance_lost", msg));
                }
                Err(e) => {
                    // Said once per silence: the run's log shows why its numbers stand still.
                    if unreachable.is_none() {
                        // Said on the run too, so a page shows the run's numbers stand still and why. A silence a
                        // process before this one recorded goes on: it keeps its start. (Before the log line, so a
                        // reader of the log finds the run saying so.)
                        let now = rfc3339(unix_now());
                        self.entry.update_cloud(|c| {
                            c.unanswered_since.get_or_insert(now);
                        });
                        self.entry.log(
                            LogLevel::Warn,
                            format!("the worker did not answer ({e}); the run goes on there, asking again"),
                        );
                    }
                    let since = *unreachable.get_or_insert_with(Instant::now);
                    wait = (wait * 2).min(t.poll_max);
                    let lost = match self.instance_state() {
                        Some(state) => {
                            if state.is_lost() {
                                gone.get_or_insert_with(Instant::now);
                            } else {
                                gone = None;
                            }
                            gone.is_some_and(|g| g.elapsed() >= t.lost_grace)
                        }
                        None => self.private.instance_id.is_none() && since.elapsed() >= t.lost_grace,
                    };
                    if lost {
                        if self.instance_ended() {
                            return Err(self.lost_by_hand());
                        }
                        if self.c.provider.info().provisions
                            && stop_requested_at.is_none()
                            && !self.is_stopping()
                            && !self.past_deadline()
                        {
                            return Err(Halt::Preempted(format!(
                                "the worker stopped answering ({e}) and its instance is gone"
                            )));
                        }
                        return Err(failed(
                            "cloud_instance_lost",
                            format!(
                                "the worker stopped answering ({e}) and its instance is gone; the weights copied so \
                                 far stay (resume the run to continue from them)"
                            ),
                        ));
                    }
                }
            }
            if cost_at.elapsed() >= COST_EVERY {
                self.update_cost();
                cost_at = Instant::now();
            }
            self.wait(wait)?;
        };
        self.collect(client, &run_id, end, time_up)
    }

    /// Where the provider says the instance is (`None` for a `remote` server, or when the provider does not answer).
    pub(super) fn instance_state(&self) -> Option<InstanceState> {
        let id = self.private.instance_id.as_ref()?;
        let provider_ref = self.core.ledger.get(id).ok()?.provider_ref?;
        self.c.provider.state(&self.c.connection.values(), &provider_ref).ok()
    }

    /// The run's instance was terminated by someone else (by hand, `DELETE /api/cloud/instances/{id}`).
    pub(super) fn instance_ended(&self) -> bool {
        let Some(id) = &self.private.instance_id else { return false };
        self.core.ledger.get(id).is_ok_and(|i| {
            matches!(
                i.status,
                CloudInstanceStatus::Terminating
                    | CloudInstanceStatus::Terminated
                    | CloudInstanceStatus::Lost
            )
        })
    }

    pub(super) fn lost_by_hand(&self) -> Halt {
        failed(
            "cloud_instance_lost",
            "the instance was terminated by hand before the run ended; the weights copied so far stay (resume the \
             run to continue from them)",
        )
    }

    fn update_cost(&self) {
        if let Some(cost) = self.cost() {
            self.entry.update_cloud(|c| c.cost_usd = Some(cost));
        }
    }

    /// Asks the worker to stop its run (retried up to `attempts` while before `deadline`; a run that ended
    /// meanwhile is fine). Returns `Ok(true)` once the worker acknowledged the stop or reported `run_not_active`,
    /// `Ok(false)` if delivery failed.
    fn stop_worker_run(
        &self,
        client: &Client,
        run_id: &str,
        attempts: u32,
        deadline: Instant,
        log_error: bool,
    ) -> std::result::Result<bool, Halt> {
        for attempt in 1..=attempts {
            match client.post_empty::<Run>(&format!("/runs/{}/stop", encode(run_id))) {
                Ok(_) => return Ok(true),
                Err(e) if e.code() == "run_not_active" => return Ok(true),
                Err(e) if e.is_transient() && attempt < attempts => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        if log_error {
                            self.entry.log(LogLevel::Warn, format!("asking the worker to stop the run: {e}"));
                        }
                        return Ok(false);
                    }
                    self.wait(self.core.tuning.poll.min(remaining))?;
                }
                Err(e) => {
                    if log_error {
                        self.entry.log(LogLevel::Warn, format!("asking the worker to stop the run: {e}"));
                    }
                    return Ok(false);
                }
            }
        }
        Ok(false)
    }
}
