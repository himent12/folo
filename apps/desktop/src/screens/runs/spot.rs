//! Cloud spot-preemption automatic checkpoint recovery status derived from run state,
//! `folo-server` preemption log lines, run error messages, and the cloud instance ledger.

use crate::api::{CloudInstance, CloudPhase, LogLevel, LogLine, Run, RunCloud, RunStatus};
use crate::ui::Tone;

/// Default automatic preemption retry budget in `folo-server` (`MAX_PREEMPTION_RETRIES = 3`,
/// `docs/CLOUD.md`).
pub const DEFAULT_MAX_PREEMPTION_RETRIES: u32 = 3;

/// High-level state of cloud spot preemption auto-resume for a run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpotRecoveryState {
    /// Worker was preempted/lost and the server is actively re-provisioning, syncing, or backing off.
    Recovering,
    /// One or more preemption recoveries succeeded and training resumed (or finished cleanly).
    Recovered,
    /// Replacement worker failed during provisioning, boot, or sync before training resumed.
    RecoveryFailed,
    /// Run failed after exhausting all automatic checkpoint resume attempts.
    Exhausted,
    /// Run failed after preemption because no resumable synced checkpoint (`weights/last.folo`) existed.
    NoCheckpoint,
    /// Run was stopped (by user or time limit) while preemption recovery was in progress.
    StoppedDuringRecovery,
    /// Active run on a provisioning provider with a synced checkpoint ready for auto-resume.
    Armed,
    /// Active run on a provisioning provider before its first synced checkpoint.
    WaitingForCheckpoint,
    /// Completed/stopped/failed run on a provisioning provider where preemption never triggered.
    NotUsed,
    /// Non-provisioning provider (e.g. remote FOLO server): no cloud instance auto-replacement.
    Unsupported,
}

/// Structured summary of cloud spot preemption auto-resume state for a cloud run.
#[derive(Clone, Debug, PartialEq)]
pub struct SpotRecoveryStatus {
    pub state: SpotRecoveryState,
    /// Number of preemption recovery attempts observed in the current run attempt.
    pub retries: u32,
    /// Maximum automatic retry budget (parsed from server messages when available, else default 3).
    pub max_retries: u32,
    /// Count of prior replaced instances for this run attempt in `/api/cloud/instances`.
    pub replaced_instances: u32,
    /// Checkpoint epoch being resumed from, when reported by `folo-server` logs.
    pub resume_epoch: Option<u32>,
    /// Backoff delay in seconds before provisioning the replacement instance, if reported.
    pub backoff_s: Option<f32>,
    /// Short preemption cause from the latest server warning, when available.
    pub reason: Option<String>,
    /// True when `folo-server` logged that it restarted mid-recovery and continued auto-resume.
    pub server_restarted_during_recovery: bool,
}

impl SpotRecoveryStatus {
    /// Compact value shown in the Cloud panel's "Spot auto-resume" fact cell.
    pub fn fact_value(&self) -> String {
        match self.state {
            SpotRecoveryState::Recovering => {
                let attempt = self.retries.max(1);
                match self.resume_epoch {
                    Some(ep) => format!("Recovering · {attempt}/{} (ep {ep})", self.max_retries),
                    None => format!("Recovering · {attempt}/{}", self.max_retries),
                }
            }
            SpotRecoveryState::Recovered => {
                let used = self.retries.max(1);
                format!("Recovered · {used}/{} used", self.max_retries)
            }
            SpotRecoveryState::RecoveryFailed => {
                let used = self.retries.max(1);
                format!("Recovery failed · {used}/{}", self.max_retries)
            }
            SpotRecoveryState::Exhausted => {
                let used = self.retries.max(self.max_retries);
                format!("Exhausted · {used}/{}", self.max_retries)
            }
            SpotRecoveryState::NoCheckpoint => "No checkpoint yet".into(),
            SpotRecoveryState::StoppedDuringRecovery => {
                let used = self.retries.max(1);
                format!("Stopped · {used}/{}", self.max_retries)
            }
            SpotRecoveryState::Armed => format!("Armed · 0/{} used", self.max_retries),
            SpotRecoveryState::WaitingForCheckpoint => format!("Waiting for ckpt · 0/{}", self.max_retries),
            SpotRecoveryState::NotUsed => format!("0/{} used", self.max_retries),
            SpotRecoveryState::Unsupported => "Not applicable".into(),
        }
    }

    /// Tooltip explaining the spot auto-resume budget and current recovery state.
    pub fn fact_tooltip(&self) -> String {
        match self.state {
            SpotRecoveryState::Recovering => format!(
                "Worker preemption detected; folo serve is automatically replacing the instance and resuming from weights/last.folo (retry budget: {}/{}).",
                self.retries.max(1),
                self.max_retries
            ),
            SpotRecoveryState::Recovered => format!(
                "Automatically recovered from {} spot preemption{} ({}/{} retry budget used).",
                self.retries.max(1),
                if self.retries.max(1) == 1 { "" } else { "s" },
                self.retries.max(1),
                self.max_retries
            ),
            SpotRecoveryState::RecoveryFailed => format!(
                "Replacement instance failed before training resumed (attempt {}/{}). Synced checkpoint weights remain on disk — use Resume to continue manually.",
                self.retries.max(1),
                self.max_retries
            ),
            SpotRecoveryState::Exhausted => format!(
                "Automatic checkpoint resume used all {} retry attempt(s). Synced weights are preserved on disk — use Resume to continue manually.",
                self.max_retries
            ),
            SpotRecoveryState::NoCheckpoint => {
                "Worker was preempted without a resumable synced checkpoint (weights/last.folo), so automatic resume could not continue.".into()
            }
            SpotRecoveryState::StoppedDuringRecovery => {
                "Run ended while automatic preemption recovery was underway; the last synced checkpoint stays on disk.".into()
            }
            SpotRecoveryState::Armed => format!(
                "A synced checkpoint is on disk; if the spot instance is preempted, folo serve auto-resumes up to {} times.",
                self.max_retries
            ),
            SpotRecoveryState::WaitingForCheckpoint => format!(
                "Auto-resume (up to {} retries) activates once the first checkpoint copies back to this server.",
                self.max_retries
            ),
            SpotRecoveryState::NotUsed => "No spot preemption occurred during this run.".into(),
            SpotRecoveryState::Unsupported => {
                "Remote FOLO servers do not provision replacement instances automatically.".into()
            }
        }
    }

    /// Optional badge shown beside the run's status badge in the Run Detail header.
    pub fn header_badge(&self) -> Option<(&'static str, String, Tone)> {
        match self.state {
            SpotRecoveryState::Recovering => Some((
                "icons/refresh-cw.svg",
                format!("Auto-resuming {}/{}", self.retries.max(1), self.max_retries),
                Tone::Warning,
            )),
            SpotRecoveryState::Recovered => Some((
                "icons/refresh-cw.svg",
                format!("Spot recovered ({}/{})", self.retries.max(1), self.max_retries),
                Tone::Info,
            )),
            SpotRecoveryState::RecoveryFailed => Some((
                "icons/triangle-alert.svg",
                format!("Spot recovery failed ({}/{})", self.retries.max(1), self.max_retries),
                Tone::Danger,
            )),
            SpotRecoveryState::Exhausted => Some((
                "icons/triangle-alert.svg",
                format!(
                    "Spot retries exhausted ({}/{})",
                    self.retries.max(self.max_retries),
                    self.max_retries
                ),
                Tone::Danger,
            )),
            _ => None,
        }
    }

    /// Optional status banner rendered inside the Cloud Run panel when preemption recovery is
    /// active, completed, stopped, or failed.
    pub fn banner(&self) -> Option<(&'static str, String, Tone)> {
        let ep_note = self.resume_epoch.map_or(String::new(), |ep| format!(" from epoch {ep}"));
        let backoff_note =
            self.backoff_s.filter(|s| *s > 0.0).map_or(String::new(), |s| format!(" after {s:.1}s backoff"));
        let cause_note =
            self.reason.as_deref().filter(|s| !s.is_empty()).map_or(String::new(), |r| format!(" ({r})"));
        let replaced_note = if self.replaced_instances > 0 {
            format!(
                " · {} prior instance{} replaced",
                self.replaced_instances,
                if self.replaced_instances == 1 { "" } else { "s" }
            )
        } else {
            String::new()
        };
        match self.state {
            SpotRecoveryState::Recovering => {
                let attempt = self.retries.max(1);
                let restart = if self.server_restarted_during_recovery {
                    " Server restarted mid-recovery and is continuing auto-resume."
                } else {
                    ""
                };
                Some((
                    "icons/refresh-cw.svg",
                    format!(
                        "Spot preemption recovery in progress (attempt {attempt}/{}{backoff_note}): replacing instance and resuming{ep_note} from synced checkpoint weights/last.folo{cause_note}.{restart}{replaced_note}",
                        self.max_retries
                    ),
                    Tone::Warning,
                ))
            }
            SpotRecoveryState::Recovered => {
                let used = self.retries.max(1);
                Some((
                    "icons/refresh-cw.svg",
                    format!(
                        "Recovered from spot preemption{cause_note}: resumed{ep_note} from weights/last.folo ({used}/{} automatic retries used{backoff_note}).{replaced_note}",
                        self.max_retries
                    ),
                    Tone::Info,
                ))
            }
            SpotRecoveryState::RecoveryFailed => {
                let used = self.retries.max(1);
                Some((
                    "icons/triangle-alert.svg",
                    format!(
                        "Spot preemption recovery failed (attempt {used}/{}): replacement worker did not reach training{cause_note}. Synced checkpoint weights stay on disk — resume the run to try again.{replaced_note}",
                        self.max_retries
                    ),
                    Tone::Danger,
                ))
            }
            SpotRecoveryState::Exhausted => Some((
                "icons/triangle-alert.svg",
                format!(
                    "Spot auto-resume exhausted all {}/{} retry attempt(s){cause_note}. Synced checkpoint weights stay on disk — resume the run to continue.{replaced_note}",
                    self.retries.max(self.max_retries),
                    self.max_retries
                ),
                Tone::Danger,
            )),
            SpotRecoveryState::NoCheckpoint => Some((
                "icons/triangle-alert.svg",
                format!(
                    "Worker was preempted{cause_note} without a resumable synced checkpoint (weights/last.folo unavailable), so automatic resume could not continue."
                ),
                Tone::Warning,
            )),
            SpotRecoveryState::StoppedDuringRecovery => Some((
                "icons/stop.svg",
                format!(
                    "Stopped during spot preemption recovery (attempt {}/{}). Synced checkpoint weights/last.folo kept on disk.",
                    self.retries.max(1),
                    self.max_retries
                ),
                Tone::Neutral,
            )),
            SpotRecoveryState::Armed
            | SpotRecoveryState::WaitingForCheckpoint
            | SpotRecoveryState::NotUsed
            | SpotRecoveryState::Unsupported => None,
        }
    }

    /// Targeted hint shown in Run Detail's failure banner when a cloud run failed due to preemption.
    pub fn failure_hint(&self) -> Option<String> {
        match self.state {
            SpotRecoveryState::RecoveryFailed => Some(
                "Replacement instance provisioning or startup failed during automatic spot recovery. Synced weights stay on disk — click Resume to try again."
                    .into(),
            ),
            SpotRecoveryState::Exhausted => Some(format!(
                "Automatic spot recovery exhausted its {}-retry budget. Synced weights stay on disk — click Resume to launch a fresh instance from the last checkpoint.",
                self.max_retries
            )),
            SpotRecoveryState::NoCheckpoint => Some(
                "The spot instance was lost without a resumable synced checkpoint. Use New run from this config to start again, or pick an on-demand offer."
                    .into(),
            ),
            _ => None,
        }
    }
}

#[derive(Default)]
struct ParsedPreemptLog {
    attempt: Option<u32>,
    max_retries: Option<u32>,
    resume_epoch: Option<u32>,
    backoff_s: Option<f32>,
    reason: Option<String>,
    restarted_during_recovery: bool,
    /// True once a replacement worker confirmed starting training after the latest preemption event.
    confirmed_resumed_after_preempt: bool,
    /// True when the current run attempt logged a stop or deadline expiry during recovery.
    stopped_during_recovery_log: bool,
    /// Timestamp (`RFC3339` string) of the latest manual resume marker (`resuming from weights/last.folo`
    /// or `no epoch finished before; starting over`) so ledger rows from earlier attempts are excluded.
    manual_resume_t: Option<String>,
}

const PREEMPT_SPLIT: &str = "); auto-resuming from synced checkpoint";

/// Parses `folo-server`'s preemption auto-resume warning lines (`crates/folo-server/src/cloud/executor/mod.rs`),
/// resetting state at manual-resume markers so an earlier attempt's warnings never bleed into a new attempt.
fn parse_preempt_logs(logs: &[LogLine]) -> ParsedPreemptLog {
    let mut out = ParsedPreemptLog::default();
    for line in logs {
        if !matches!(line.level, LogLevel::Warn | LogLevel::Error | LogLevel::Info) {
            continue;
        }
        let msg = line.message.as_str();
        // Manual resume (`POST /api/runs` with `resume_run_id`) resets preemption history for the new attempt.
        if msg == "resuming from weights/last.folo" || msg == "no epoch finished before; starting over" {
            out = ParsedPreemptLog { manual_resume_t: Some(line.t.clone()), ..Default::default() };
            continue;
        }
        if msg.contains("server restarted during preemption recovery") {
            out.restarted_during_recovery = true;
            out.confirmed_resumed_after_preempt = false;
            continue;
        }
        let lower = msg.to_ascii_lowercase();
        if lower.contains("during preemption recovery")
            || lower.contains("before a replacement worker could launch")
            || lower.contains("ended the run after worker preemption")
        {
            out.stopped_during_recovery_log = true;
            continue;
        }
        if let Some((prefix, suffix)) = msg.split_once(PREEMPT_SPLIT) {
            out.confirmed_resumed_after_preempt = false;
            out.stopped_during_recovery_log = false;
            if let Some(why) = prefix.strip_prefix("worker preempted or lost (") {
                let trimmed = why.trim();
                if !trimmed.is_empty() {
                    out.reason = Some(trimmed.to_string());
                }
            }
            if let Some((_, rest)) = suffix.split_once("epoch ") {
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(ep) = digits.parse::<u32>() {
                    out.resume_epoch = Some(ep);
                }
            }
            if let Some((_, rest)) = suffix.split_once("attempt ") {
                let a_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if let Ok(a) = a_str.parse::<u32>() {
                    out.attempt = Some(out.attempt.map_or(a, |prev| prev.max(a)));
                    if let Some(after_slash) = rest.get(a_str.len()..).and_then(|s| s.strip_prefix('/')) {
                        let m_str: String = after_slash.chars().take_while(|c| c.is_ascii_digit()).collect();
                        if let Ok(m) = m_str.parse::<u32>()
                            && m > 0
                        {
                            out.max_retries = Some(m);
                        }
                    }
                }
            }
            if let Some((_, rest)) = suffix.split_once("after ") {
                let num: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
                if let Ok(s) = num.parse::<f32>()
                    && s.is_finite()
                {
                    out.backoff_s = Some(s);
                }
            } else {
                out.backoff_s = None;
            }
            continue;
        }
        // A replacement worker reaching training confirms recovery succeeded for this preemption.
        if (out.attempt.is_some() || out.restarted_during_recovery)
            && (msg.starts_with("the worker took the run as ") || msg.starts_with("epoch "))
        {
            out.confirmed_resumed_after_preempt = true;
        }
    }
    out
}

/// Parses `"automatic checkpoint resume exhausted all N attempt(s)"` from `Run::error`.
fn parse_exhausted_max(error: &str) -> Option<u32> {
    let (_, rest) = error.split_once("exhausted all ")?;
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().ok().filter(|&n| n > 0)
}

/// Computes the spot preemption auto-resume status of cloud run `r` (`c = r.cloud`).
pub fn spot_recovery_status(
    r: &Run,
    c: &RunCloud,
    provider_provisions: bool,
    logs: Option<&[LogLine]>,
    instances: Option<&[CloudInstance]>,
) -> SpotRecoveryStatus {
    if !provider_provisions {
        return SpotRecoveryStatus {
            state: SpotRecoveryState::Unsupported,
            retries: 0,
            max_retries: 0,
            replaced_instances: 0,
            resume_epoch: None,
            backoff_s: None,
            reason: None,
            server_restarted_during_recovery: false,
        };
    }

    let parsed = logs.map(parse_preempt_logs).unwrap_or_default();
    let attempt_floor = parsed.manual_resume_t.as_deref();
    let replaced_instances = instances.map_or(0, |list| {
        list.iter()
            .filter(|i| i.run_id.as_deref() == Some(r.id.as_str()))
            .filter(|i| c.instance_id.as_deref() != Some(i.id.as_str()))
            .filter(|i| attempt_floor.is_none_or(|t| i.created.as_str() >= t))
            .count() as u32
    });
    let err = r.error.as_deref().unwrap_or("");
    let err_lower = err.to_ascii_lowercase();
    let exhausted_max = parse_exhausted_max(&err_lower);
    let max_retries = exhausted_max.or(parsed.max_retries).unwrap_or(DEFAULT_MAX_PREEMPTION_RETRIES);

    let err_exhausted = r.status == RunStatus::Failed
        && (exhausted_max.is_some() || err_lower.contains("automatic checkpoint resume exhausted"));
    let err_no_ckpt = r.status == RunStatus::Failed
        && ((err_lower.contains("no synced checkpoint") && err_lower.contains("auto-resume"))
            || err_lower.contains("no incomplete training state to auto-resume")
            || err_lower.contains("no incomplete weights/last.folo"));
    // Only count prior ledger instances as spot replacements when preemption/recovery evidence is
    // present in the current attempt's logs or error: a manually resumed cloud run shares the same
    // run ID in the ledger and its manual-resume log marker may be unloaded or scrolled out of the
    // 1,000-line tail window.
    let has_preempt_evidence = parsed.attempt.is_some()
        || parsed.restarted_during_recovery
        || parsed.stopped_during_recovery_log
        || err_exhausted
        || err_no_ckpt
        || err_lower.contains("during preemption recovery");
    let replaced_instances = if has_preempt_evidence { replaced_instances } else { 0 };
    let retries = parsed.attempt.unwrap_or(0).max(replaced_instances).max(exhausted_max.unwrap_or(0));

    let has_recovery_activity = retries > 0 || parsed.restarted_during_recovery;
    let ended_before_training = c.ended_in.is_some_and(|p| p < CloudPhase::Training);
    let unconfirmed_replacement = logs.is_some()
        && (parsed.attempt.is_some() || parsed.restarted_during_recovery)
        && !parsed.confirmed_resumed_after_preempt;
    let releasing_after_failed_recovery = r.status.is_active()
        && c.phase >= CloudPhase::Releasing
        && has_recovery_activity
        && (ended_before_training || unconfirmed_replacement);
    let stopped_in_recovery = (r.status == RunStatus::Stopped
        && (parsed.stopped_during_recovery_log
            || err_lower.contains("during preemption recovery")
            || (has_recovery_activity && ended_before_training)))
        || (releasing_after_failed_recovery && parsed.stopped_during_recovery_log);
    let failed_in_recovery = (r.status == RunStatus::Failed
        && has_recovery_activity
        && (ended_before_training || unconfirmed_replacement))
        || (releasing_after_failed_recovery && !parsed.stopped_during_recovery_log);

    let state = if err_exhausted {
        SpotRecoveryState::Exhausted
    } else if err_no_ckpt {
        SpotRecoveryState::NoCheckpoint
    } else if stopped_in_recovery {
        SpotRecoveryState::StoppedDuringRecovery
    } else if failed_in_recovery {
        SpotRecoveryState::RecoveryFailed
    } else if has_recovery_activity {
        if r.status.is_active() && c.phase < CloudPhase::Training {
            SpotRecoveryState::Recovering
        } else {
            SpotRecoveryState::Recovered
        }
    } else if r.status.is_active() {
        if c.synced.is_some() { SpotRecoveryState::Armed } else { SpotRecoveryState::WaitingForCheckpoint }
    } else {
        SpotRecoveryState::NotUsed
    };

    SpotRecoveryStatus {
        state,
        retries,
        max_retries,
        replaced_instances,
        resume_epoch: parsed.resume_epoch,
        backoff_s: parsed.backoff_s,
        reason: parsed.reason,
        server_restarted_during_recovery: parsed.restarted_during_recovery,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{Backend, CloudInstanceStatus, mock::MockBackend};

    fn cloud_run_fixture() -> (Run, RunCloud) {
        let api = MockBackend::with_cloud_runs(0, true);
        let r = api.run("run_c2e1").unwrap();
        let c = r.cloud.clone().unwrap();
        (r, c)
    }

    fn log(level: LogLevel, message: &str) -> LogLine {
        LogLine { t: "2026-11-08T07:20:00Z".into(), level, message: message.into() }
    }

    fn log_at(t: &str, level: LogLevel, message: &str) -> LogLine {
        LogLine { t: t.into(), level, message: message.into() }
    }

    #[test]
    fn armed_waiting_and_unsupported_states_follow_checkpoint_and_provider() {
        let (mut r, mut c) = cloud_run_fixture();
        let remote = spot_recovery_status(&r, &c, false, None, None);
        assert_eq!(remote.state, SpotRecoveryState::Unsupported);
        assert_eq!(remote.fact_value(), "Not applicable");
        assert!(remote.banner().is_none() && remote.header_badge().is_none());

        c.synced = Some("2026-11-08T07:20:00Z".into());
        let armed = spot_recovery_status(&r, &c, true, Some(&[]), Some(&[]));
        assert_eq!(armed.state, SpotRecoveryState::Armed);
        assert_eq!(armed.fact_value(), "Armed · 0/3 used");
        assert!(armed.banner().is_none());

        c.synced = None;
        let waiting = spot_recovery_status(&r, &c, true, Some(&[]), Some(&[]));
        assert_eq!(waiting.state, SpotRecoveryState::WaitingForCheckpoint);
        assert!(waiting.fact_value().contains("0/3"));

        r.status = RunStatus::Completed;
        c.phase = CloudPhase::Released;
        let done = spot_recovery_status(&r, &c, true, Some(&[]), Some(&[]));
        assert_eq!(done.state, SpotRecoveryState::NotUsed);
        assert_eq!(done.fact_value(), "0/3 used");
    }

    #[test]
    fn parses_server_preemption_logs_ignoring_keywords_inside_provider_reason() {
        let (mut r, mut c) = cloud_run_fixture();
        let logs = vec![log(
            LogLevel::Warn,
            "worker preempted or lost (attempt 9/9 after 99s epoch 999); auto-resuming from synced checkpoint weights/last.folo (epoch 14, attempt 2/3 after 10.0s backoff)",
        )];
        r.status = RunStatus::Queued;
        c.phase = CloudPhase::Syncing;
        let st = spot_recovery_status(&r, &c, true, Some(&logs), None);
        assert_eq!(st.state, SpotRecoveryState::Recovering);
        assert_eq!(st.retries, 2);
        assert_eq!(st.max_retries, 3);
        assert_eq!(st.resume_epoch, Some(14));
        assert_eq!(st.backoff_s, Some(10.0));
        assert_eq!(st.reason.as_deref(), Some("attempt 9/9 after 99s epoch 999"));
        assert!(
            st.header_badge()
                .is_some_and(|(_, label, tone)| label == "Auto-resuming 2/3" && tone == Tone::Warning)
        );

        r.status = RunStatus::Running;
        c.phase = CloudPhase::Training;
        let rec = spot_recovery_status(&r, &c, true, Some(&logs), None);
        assert_eq!(rec.state, SpotRecoveryState::Recovered);
        assert_eq!(rec.fact_value(), "Recovered · 2/3 used");
        assert!(
            rec.header_badge()
                .is_some_and(|(_, label, tone)| label == "Spot recovered (2/3)" && tone == Tone::Info)
        );
    }

    #[test]
    fn failed_or_deadline_stopped_replacement_never_reads_as_recovered_and_manual_resume_resets() {
        let (mut r, mut c) = cloud_run_fixture();
        let logs = vec![log_at(
            "2026-11-08T06:00:00Z",
            LogLevel::Warn,
            "worker preempted or lost (spot reclaimed); auto-resuming from synced checkpoint weights/last.folo (epoch 12, attempt 1/3 after 10.0s backoff)",
        )];

        // 1. Replacement failed while provisioning -> RecoveryFailed (never Recovered!).
        r.status = RunStatus::Failed;
        r.error_code = Some("cloud_provider_error".into());
        r.error = Some("RunPod out of stock".into());
        c.phase = CloudPhase::Released;
        c.ended_in = Some(CloudPhase::Provisioning);
        let failed_rec = spot_recovery_status(&r, &c, true, Some(&logs), None);
        assert_eq!(failed_rec.state, SpotRecoveryState::RecoveryFailed);
        assert!(failed_rec.header_badge().is_some_and(|(_, _, tone)| tone == Tone::Danger));
        assert!(failed_rec.failure_hint().is_some());

        // 1b. Still-active Releasing phase after replacement failed during provisioning -> RecoveryFailed.
        r.status = RunStatus::Running;
        c.phase = CloudPhase::Releasing;
        c.ended_in = Some(CloudPhase::Provisioning);
        let releasing_failed = spot_recovery_status(&r, &c, true, Some(&logs), None);
        assert_eq!(releasing_failed.state, SpotRecoveryState::RecoveryFailed);

        // 2. Time limit expired during preemption recovery -> StoppedDuringRecovery (never Recovered!).
        r.status = RunStatus::Stopped;
        r.error = None;
        c.ended_in = Some(CloudPhase::Provisioning);
        let mut deadline_logs = logs.clone();
        deadline_logs.push(log_at(
            "2026-11-08T06:01:00Z",
            LogLevel::Warn,
            "the time limit of 4 h ended the run during preemption recovery",
        ));
        let stopped_dl = spot_recovery_status(&r, &c, true, Some(&deadline_logs), None);
        assert_eq!(stopped_dl.state, SpotRecoveryState::StoppedDuringRecovery);

        // 3. Server restart with no incomplete checkpoint -> NoCheckpoint.
        r.status = RunStatus::Failed;
        r.error = Some(
            "server restarted during preemption recovery, but no incomplete weights/last.folo is available to resume from"
                .into(),
        );
        let no_inc = spot_recovery_status(&r, &c, true, Some(&logs), None);
        assert_eq!(no_inc.state, SpotRecoveryState::NoCheckpoint);

        // 4. Manual resume resets prior attempt's preemption logs and excludes older ledger instances.
        let (mut r3, mut c3) = cloud_run_fixture();
        r3.status = RunStatus::Queued;
        c3.phase = CloudPhase::Provisioning;
        c3.synced = Some("2026-11-08T06:00:00Z".into());
        let mut resumed_logs = logs.clone();
        resumed_logs.push(log_at("2026-11-08T07:00:00Z", LogLevel::Info, "resuming from weights/last.folo"));
        let old_instance = CloudInstance {
            id: "ci_old_attempt".into(),
            connection_id: c3.connection_id.clone(),
            provider: c3.provider.clone(),
            provider_ref: Some("pod-old".into()),
            name: "folo-old".into(),
            run_id: Some(r3.id.clone()),
            offer: c3.offer.clone(),
            status: CloudInstanceStatus::Terminated,
            created: "2026-11-08T05:30:00Z".into(),
            ended: Some("2026-11-08T06:00:00Z".into()),
            deadline: None,
            cost_usd: Some(0.50),
            error: None,
        };
        let fresh_attempt = spot_recovery_status(&r3, &c3, true, Some(&resumed_logs), Some(&[old_instance]));
        assert_eq!(fresh_attempt.state, SpotRecoveryState::Armed);
        assert_eq!(fresh_attempt.retries, 0);
        assert_eq!(fresh_attempt.replaced_instances, 0);
    }

    #[test]
    fn detects_exhausted_no_checkpoint_and_ledger_only_replacement() {
        let (mut r, mut c) = cloud_run_fixture();
        r.status = RunStatus::Failed;
        r.error_code = Some("cloud_instance_lost".into());
        r.error = Some(
            "worker terminated by provider; automatic checkpoint resume exhausted all 4 attempt(s) — the weights copied so far stay"
                .into(),
        );
        c.phase = CloudPhase::Released;
        let ex = spot_recovery_status(&r, &c, true, None, None);
        assert_eq!(ex.state, SpotRecoveryState::Exhausted);
        assert_eq!(ex.retries, 4);
        assert_eq!(ex.max_retries, 4);

        r.error = Some(
            "spot reclaimed; the synced weights/last.folo has no incomplete training state to auto-resume from".into(),
        );
        let no_ckpt = spot_recovery_status(&r, &c, true, None, None);
        assert_eq!(no_ckpt.state, SpotRecoveryState::NoCheckpoint);

        // Prior ledger rows without preemption log/error evidence must NOT fabricate a Recovered
        // badge (prevents manual-resume false positives when logs are unloaded or truncated).
        let (r2, mut c2) = cloud_run_fixture();
        c2.synced = Some("2026-11-08T06:05:00Z".into());
        let replaced = CloudInstance {
            id: "ci_old_spot".into(),
            connection_id: c2.connection_id.clone(),
            provider: c2.provider.clone(),
            provider_ref: Some("pod-old".into()),
            name: "folo-old".into(),
            run_id: Some(r2.id.clone()),
            offer: c2.offer.clone(),
            status: CloudInstanceStatus::Terminated,
            created: "2026-11-08T05:00:00Z".into(),
            ended: Some("2026-11-08T06:00:00Z".into()),
            deadline: None,
            cost_usd: Some(0.69),
            error: None,
        };
        let without_logs = spot_recovery_status(&r2, &c2, true, None, Some(std::slice::from_ref(&replaced)));
        assert_eq!(without_logs.state, SpotRecoveryState::Armed);
        assert_eq!(without_logs.replaced_instances, 0);
        assert_eq!(without_logs.retries, 0);

        // With preemption log evidence present, ledger replacement rows are counted.
        let preempt_log = vec![log_at(
            "2026-11-08T06:01:00Z",
            LogLevel::Warn,
            "worker preempted or lost (reclaimed); auto-resuming from synced checkpoint weights/last.folo (epoch 4, attempt 1/3 after 10.0s backoff)",
        )];
        let with_logs = spot_recovery_status(&r2, &c2, true, Some(&preempt_log), Some(&[replaced]));
        assert_eq!(with_logs.state, SpotRecoveryState::Recovered);
        assert_eq!(with_logs.replaced_instances, 1);
        assert_eq!(with_logs.retries, 1);
    }
}
