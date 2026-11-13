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
        let mut stop_sent: Option<Instant> = None;
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
            if stop_sent.is_none() && (stopping || self.past_deadline()) {
                time_up = !stopping;
                if time_up {
                    self.entry.log(
                        LogLevel::Warn,
                        format!("the time limit of {} is up; stopping the worker's run", self.hours()),
                    );
                }
                self.stop_worker_run(client, &run_id);
                stop_sent = Some(Instant::now());
            }
            if stop_sent.is_some_and(|s| s.elapsed() >= t.lost_grace) {
                // A worker that has not answered for most of that time has nothing to collect from: the run is lost
                // with its instance (a time limit that ended while nothing followed the run, and the instance
                // with it). Half the grace, not all of it: the silence is noticed a poll after the stop was sent.
                if unreachable.is_some_and(|u| u.elapsed() >= t.lost_grace / 2) {
                    return Err(failed(
                        "cloud_instance_lost",
                        "the worker stopped answering before the run could be stopped and collected; the weights \
                         copied so far stay (resume the run to continue from them)",
                    ));
                }
                self.entry
                    .log(LogLevel::Warn, "the worker's run did not stop in time; collecting what it has");
                break None;
            }
            match self.mirror(client, &run_id) {
                Ok(Some(end)) => break Some(end),
                Ok(None) => {
                    if unreachable.is_some() {
                        self.entry.log(LogLevel::Info, "the worker answers again");
                        self.entry.update_cloud(|c| c.unanswered_since = None);
                    }
                    // A run followed again whose record does not say when its worker first answered (a server
                    // of before wrote it, or that write failed): it answers now. The first time stays.
                    if !std::mem::replace(&mut answered, true)
                        && let Some(id) = &self.private.instance_id
                    {
                        let _ = self.core.ledger.booted(id);
                    }
                    (wait, unreachable, gone) = (t.poll, None, None);
                }
                Err(e) if e.code() == "run_not_found" => {
                    return Err(failed(
                        "cloud_instance_lost",
                        "the worker no longer has the run (its server restarted or was replaced)",
                    ));
                }
                Err(e) => {
                    // Said once per silence: the run's log shows why its numbers stand still.
                    if unreachable.is_none() {
                        self.entry.log(
                            LogLevel::Warn,
                            format!("the worker did not answer ({e}); the run goes on there, asking again"),
                        );
                        // Said on the run too, so a page shows the run's numbers stand still and why.
                        let now = rfc3339(unix_now());
                        self.entry.update_cloud(|c| c.unanswered_since = Some(now));
                    }
                    let since = *unreachable.get_or_insert_with(Instant::now);
                    wait = (wait * 2).min(t.poll_max);
                    let lost = match self.instance_state() {
                        Some(state) => {
                            if matches!(state, InstanceState::Gone | InstanceState::Stopped) {
                                gone.get_or_insert_with(Instant::now);
                            } else {
                                gone = None;
                            }
                            gone.is_some_and(|g| g.elapsed() >= t.lost_grace)
                        }
                        None => self.private.instance_id.is_none() && since.elapsed() >= t.lost_grace,
                    };
                    if lost {
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
    fn instance_state(&self) -> Option<InstanceState> {
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

    /// Asks the worker to stop its run (retried; a run that ended meanwhile is fine).
    fn stop_worker_run(&self, client: &Client, run_id: &str) {
        for attempt in 1..=ATTEMPTS {
            match client.post_empty::<Run>(&format!("/runs/{}/stop", encode(run_id))) {
                Ok(_) => return,
                Err(e) if e.code() == "run_not_active" => return,
                Err(e) if e.is_transient() && attempt < ATTEMPTS => {
                    if self.wait(self.core.tuning.poll).is_err() {
                        return;
                    }
                }
                Err(e) => {
                    self.entry.log(LogLevel::Warn, format!("asking the worker to stop the run: {e}"));
                    return;
                }
            }
        }
    }
}
