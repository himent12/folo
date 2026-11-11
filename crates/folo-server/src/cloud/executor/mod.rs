//! A cloud run's own thread, through the phases of docs/CLOUD.md: `provisioning` (a ledger record, then the launch),
//! `booting` (the instance, then its worker, then the worker check), `syncing` (the dataset, packed as FOLO reads it,
//! and the weights or checkpoint to start from), `training` (the worker's run, mirrored into the local run folder
//! poll by poll, its checkpoint copied back), `collecting` (the final weights), `releasing` (the instance terminated
//! through the ledger) and `released`. `Run.cloud` and a `cloud` event follow every change, and each phase gets a log
//! line a person can follow.
//!
//! Every path after an instance exists ends in its release: a [`Release`] guard does it when the run ends, fails,
//! is stopped, or panics. The only way out without it is letting go of a run that trains while this process shuts
//! down: it keeps training, and the next process follows it again from its private state ([`Private`]). A run that
//! does not train yet has nothing to lose, so a shutdown stops it and releases its instance.

mod launch;
mod mirror;
mod sync;
mod train;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::{CloudOffer, CloudPhase, LogLevel, RunStatus};
use folo_client::transfer::{self, TransferError};
use folo_cloud::InstanceState;

use super::files::{self, RunLock};
use super::state::{Outcome, Private};
use super::{Connected, Core};
use crate::error::Result;
use crate::runs::{Ended, RunEntry};
use crate::util::{fnv1a, rfc3339, unix_now};

/// Lines of `/logs` asked for per request (the API's maximum).
const LOG_PAGE: usize = 5000;
/// `Run.cloud.progress` and `cost_usd` are sent at most this often.
const PROGRESS_EVERY: Duration = Duration::from_secs(1);
const COST_EVERY: Duration = Duration::from_secs(60);
/// A long transfer logs how far it is at most this often.
const TRANSFER_LOG_EVERY: Duration = Duration::from_secs(60);
/// Tries of a worker request that changes something (start or stop its run) before giving up.
const ATTEMPTS: u32 = 5;
/// A booting instance its provider reports gone (or stopped) is believed after this many poll intervals of it.
const GONE_POLLS: u32 = 10;
/// While an instance's worker does not answer yet, its provider is asked about it every this many poll intervals.
const STATE_POLLS: u32 = 5;
/// A copy of `weights/last.folo` that failed while the run trains is tried again after this long at the earliest
/// (or after `checkpoint_every`, when that is shorter), not at every epoch.
const COPY_RETRY_EVERY: Duration = Duration::from_secs(60);
/// A step of one poll of the worker (its run, its history, its samples, its log) that takes this long is written
/// to the server's log: what held the live view up.
const SLOW_STEP: Duration = Duration::from_secs(5);

/// A cloud run to execute.
pub(crate) struct Job {
    /// The registered run, claimed for this job.
    pub entry: Arc<RunEntry>,
    /// The connection it computes on.
    pub connected: Connected,
    /// The fingerprint of the dataset upload (the dataset's index fingerprint), for a run that starts; `None` when
    /// following a run again.
    pub upload: Option<String>,
}

/// Where a job starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    /// A new or resumed run: every phase from provisioning.
    Start,
    /// A run a previous process left active: followed again when it was training, else released and `stopped`.
    Recover,
}

/// Starts `job`'s thread, which holds `lock` until the run is released.
pub(crate) fn spawn(core: Arc<Core>, job: Job, lock: RunLock, mode: Mode) -> std::io::Result<JoinHandle<()>> {
    let name = format!("folo-cloud-{}", job.entry.id());
    std::thread::Builder::new().name(name).spawn(move || {
        let entry = job.entry.clone();
        entry.execute(|| {
            let watched = entry.clone();
            let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                Exec::new(core, job, lock).drive(mode)
            }));
            ran.unwrap_or_else(|_| {
                // The guard released the instance while the panic unwound.
                let message = "the cloud run panicked (a bug); its instance was released";
                watched.log(LogLevel::Error, message);
                watched.set_outcome(RunStatus::Failed, Some(message.into()), Some("internal".into()));
                Ended::Finished
            })
        })
    })
}

/// Ends an active cloud run a previous process left that this one cannot follow (`why`: its connection is gone):
/// `stopped`, with a warning that its instance may still run.
pub(crate) fn abandon(core: &Core, entry: &Arc<RunEntry>, why: &str) -> Result<()> {
    let dir = core.run_dir(entry.id());
    let lock = files::try_lock_run(&dir)
        .map_err(|e| {
            crate::error::Error::io(format!("locking {}: {e}", files::run_lock_path(&dir).display()))
        })?
        .ok_or_else(|| crate::runs::followed_elsewhere(entry.id()))?;
    if !entry.follow() {
        return Ok(());
    }
    entry.execute(|| {
        let _lock = lock;
        entry.log(
            LogLevel::Warn,
            format!(
                "this server cannot follow or release the cloud run: {why}; if its instance still runs, terminate it \
                 in the provider's console"
            ),
        );
        entry.set_outcome(RunStatus::Stopped, None, None);
        Ended::Finished
    });
    Ok(())
}

/// Why the phases ended early.
enum Halt {
    /// Asked to stop (or the time limit ended) before the worker trained; the reason is logged. Released at once,
    /// `stopped`.
    Stopped(String),
    /// Failed; released, `failed` with the outcome's code.
    Failed(Outcome),
    /// A previous process left the run in this phase, before it trained: released, `stopped`.
    Interrupted(CloudPhase),
    /// This process shuts down: a run that trains is left to its worker; an earlier one is released, `stopped`.
    LetGo,
}

/// `cloud_*` failures.
fn failed(code: &str, message: impl Into<String>) -> Halt {
    Halt::Failed(Outcome::failed(code, message))
}

/// What the instance's worker is reached at.
enum Endpoint {
    /// A server that already runs (`remote`).
    Fixed(folo_cloud::Endpoint),
    /// A provisioned instance, by the provider's id of it.
    Instance(String),
}

/// What syncing gave the worker.
struct Synced {
    dataset_id: String,
    /// The model to start from, and whether it is the run's checkpoint (continued) rather than weights (fine-tuned).
    model: Option<(String, bool)>,
}

/// A size for people, e.g. `1.2 MB` or `4.8 GB`.
fn size(bytes: u64) -> String {
    let b = bytes as f64;
    match bytes {
        0..1_000_000 => format!("{:.1} kB", b / 1e3),
        1_000_000..1_000_000_000 => format!("{:.1} MB", b / 1e6),
        _ => format!("{:.1} GB", b / 1e9),
    }
}

/// Disk for an instance training on a dataset that packs to `bytes`: twice the dataset plus 20 GB.
fn disk_for(bytes: u64) -> u32 {
    let gb = ((2.0 * bytes as f64 / 1e9).ceil() as u32).saturating_add(20);
    gb.clamp(*super::DISK_GB.start(), *super::DISK_GB.end())
}

/// An upload fingerprint for the file at `path`: its name, size and modification time (FOLO's writers replace
/// weights files whole, so a new version has a new time).
fn file_fingerprint(path: &Path) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let nanos = meta.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?.as_nanos();
    Some(format!("file-{:x}-{nanos:x}-{:016x}", meta.len(), fnv1a(path.to_string_lossy().as_bytes())))
}

/// The release of a run's instance, when the run ends whichever way: the phases `releasing` and `released`, the
/// termination through the ledger (waiting at most [`super::Tuning::release_wait`] for the provider to confirm, then
/// leaving the rest to the reaper) and the final cost. Done on drop unless [`Release::keep`] said the run goes on.
struct Release {
    core: Arc<Core>,
    entry: Arc<RunEntry>,
    instance: Option<String>,
    done: bool,
}

impl Release {
    fn new(core: Arc<Core>, entry: Arc<RunEntry>, instance: Option<String>) -> Release {
        Release { core, entry, instance, done: false }
    }

    /// The run has an instance now.
    fn arm(&mut self, id: &str) {
        self.instance = Some(id.to_string());
    }

    /// The launch created nothing after all.
    fn disarm(&mut self) {
        self.instance = None;
    }

    /// The run keeps its instance: this process lets go of it.
    fn keep(mut self) {
        self.done = true;
    }

    /// Releases now.
    fn finish(mut self) {
        self.release();
    }

    fn release(&mut self) {
        if std::mem::replace(&mut self.done, true) {
            return;
        }
        self.entry.update_cloud(|c| {
            c.phase = CloudPhase::Releasing;
            c.progress = None;
        });
        if let Some(id) = self.instance.take() {
            self.entry.log(LogLevel::Info, "terminating the instance");
            // A process that shuts down asks once and leaves the retries to the next one's reaper.
            let wait = if self.core.halt.is_on() { Duration::ZERO } else { self.core.tuning.release_wait };
            match self.core.terminate_within(&id, wait) {
                Ok(i) if !i.status.is_alive() => {
                    let cost = i.cost_usd.map_or(String::new(), |c| format!("; it cost about ${c:.2}"));
                    self.entry.log(LogLevel::Info, format!("terminated the instance{cost}"));
                    self.entry.update_cloud(|c| c.cost_usd = i.cost_usd);
                }
                Ok(i) => {
                    let why = i.error.as_deref().unwrap_or("the provider has not confirmed it");
                    self.entry.log(
                        LogLevel::Warn,
                        format!(
                            "the instance is not terminated yet ({why}); FOLO keeps trying in the background, see \
                             GET /api/cloud/instances or `folo cloud instances`"
                        ),
                    );
                    self.entry.update_cloud(|c| c.cost_usd = i.cost_usd);
                }
                Err(e) => self.entry.log(
                    LogLevel::Warn,
                    format!("terminating the instance: {}; check GET /api/cloud/instances", e.body.message),
                ),
            }
        }
        self.entry.update_cloud(|c| c.phase = CloudPhase::Released);
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        self.release();
    }
}

/// Since when a provider has answered that an instance is gone (or stopped), with no other answer in between.
#[derive(Default)]
struct GoneSince(Option<Instant>);

impl GoneSince {
    /// Takes the provider's answer `state`; true once gone has been the only answer for `after`. Any other
    /// answer (starting, running) starts the count over.
    fn believed(&mut self, state: &InstanceState, after: Duration) -> bool {
        match state {
            InstanceState::Stopped | InstanceState::Gone => {
                self.0.get_or_insert_with(Instant::now).elapsed() >= after
            }
            _ => {
                self.0 = None;
                false
            }
        }
    }
}

/// The file a copy of `weights/last.folo` is fetched into while the run trains ([`Copying`]), in `weights`: a
/// name of its own for every copy of this process (an older copy's thread that is still about to clean up must
/// not find a newer copy's file under its name), and one of the names [`transfer::remove_partial`] removes.
fn staged_copy(weights: &Path) -> PathBuf {
    static N: AtomicU64 = AtomicU64::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    weights.join(format!(".last.folo.copy.{}.{n}.download", std::process::id()))
}

/// A copy of the worker's `weights/last.folo` under way while the run trains, on a thread of its own: a checkpoint
/// of hundreds of megabytes takes minutes on a slow link, and the live view (the next poll of the worker), a stop
/// and the collection at the run's end must not wait for it.
///
/// The thread only ever writes its own staging file ([`staged_copy`]). The executor moves that file to
/// `weights/last.folo` when it takes the result ([`Exec::reap_copy`]), so a copy that outlives its run (a request
/// that was in flight when the run failed, or when the final file was collected) can never put an old checkpoint
/// over a newer one. Dropped, the copy is told to end, and its staged file is removed by whichever side comes
/// last.
struct Copying {
    thread: Option<JoinHandle<std::result::Result<(), TransferError>>>,
    cancel: Arc<AtomicBool>,
    staged: PathBuf,
}

impl Drop for Copying {
    fn drop(&mut self) {
        // In this order: a thread that finishes after the flag is set removes the file itself, one that finished
        // before has left it here.
        self.cancel.store(true, Ordering::SeqCst);
        let _ = std::fs::remove_file(&self.staged);
    }
}

/// One execution of a cloud run.
struct Exec {
    core: Arc<Core>,
    entry: Arc<RunEntry>,
    c: Connected,
    fingerprint: Option<String>,
    private: Private,
    dir: PathBuf,
    /// The worker's queue position last logged.
    queued_at: Option<u32>,
    /// When a copy of `weights/last.folo` last failed while the run trained ([`COPY_RETRY_EVERY`]).
    copy_failed: Option<Instant>,
    /// The copy of `weights/last.folo` under way, if one is.
    copying: Option<Copying>,
    _lock: RunLock,
}

impl Exec {
    fn new(core: Arc<Core>, job: Job, lock: RunLock) -> Exec {
        let dir = job.entry.dir().to_path_buf();
        Exec {
            core,
            private: Private::load(&dir),
            entry: job.entry,
            c: job.connected,
            fingerprint: job.upload,
            dir,
            queued_at: None,
            copy_failed: None,
            copying: None,
            _lock: lock,
        }
    }

    /// Runs the phases, releases, and records the outcome.
    fn drive(mut self, mode: Mode) -> Ended {
        // A resumed run starts over on a new instance: nothing of the last one is used.
        if mode == Mode::Start {
            self.private = Private::default();
            self.save();
        }
        // What a process that died mid-copy left next to the weights (one process follows a run at a time).
        transfer::remove_partial(&self.dir.join("weights"));
        let mut release =
            Release::new(self.core.clone(), self.entry.clone(), self.private.instance_id.clone());
        let result = match mode {
            Mode::Start => self.start(&mut release),
            Mode::Recover => self.recover(),
        };
        let outcome = match result {
            Ok(o) => o,
            Err(Halt::LetGo) if self.phase_now() >= CloudPhase::Training => {
                release.keep();
                self.save();
                self.entry.log(
                    LogLevel::Info,
                    "this server shuts down; the run keeps training on its worker, and the next `folo serve` on this \
                     data dir follows it again",
                );
                return Ended::LetGo;
            }
            // Nothing trains yet, so nothing is lost: better released now than billed until the next start.
            Err(Halt::LetGo) => {
                self.entry.log(
                    LogLevel::Warn,
                    format!(
                        "this server shuts down while the run is {}; it is stopped and its instance released (resume \
                         it with POST /api/runs and resume_run_id)",
                        self.phase_now().as_str()
                    ),
                );
                Outcome::plain(RunStatus::Stopped)
            }
            Err(Halt::Stopped(why)) => {
                self.entry.log(LogLevel::Info, why);
                Outcome::plain(RunStatus::Stopped)
            }
            Err(Halt::Failed(o)) => {
                self.entry.log(LogLevel::Error, o.error.clone().unwrap_or_default());
                o
            }
            Err(Halt::Interrupted(phase)) => {
                self.entry.log(
                    LogLevel::Warn,
                    format!(
                        "the server stopped while the run was {}; it is stopped now and its instance released \
                         (resume it with POST /api/runs and resume_run_id)",
                        phase.as_str()
                    ),
                );
                Outcome::plain(RunStatus::Stopped)
            }
        };
        self.private.outcome = Some(outcome.clone());
        self.save();
        release.finish();
        self.entry.set_outcome(outcome.status, outcome.error, outcome.error_code);
        Ended::Finished
    }

    fn save(&self) {
        self.private.save(&self.dir);
    }

    fn phase_now(&self) -> CloudPhase {
        self.entry.run().cloud.map_or(CloudPhase::Provisioning, |c| c.phase)
    }

    /// Moves to `phase` (its progress cleared, its instance's cost so far with it) and logs `message`.
    fn phase(&self, phase: CloudPhase, message: impl Into<String>) {
        let cost = self.cost();
        self.entry.update_cloud(|c| {
            c.phase = phase;
            c.progress = None;
            c.cost_usd = cost.or(c.cost_usd);
        });
        self.entry.log(LogLevel::Info, message);
    }

    /// What the run's instance costs so far, as the ledger estimates it (`None` without an instance or a price).
    fn cost(&self) -> Option<f64> {
        self.private.instance_id.as_ref().and_then(|id| self.core.ledger.get(id).ok())?.cost_usd
    }

    fn offer(&self) -> CloudOffer {
        self.entry.run().cloud.map(|c| c.offer).expect("a cloud run has its offer")
    }

    /// The run's time limit (`compute.max_hours`, or the tuning's).
    fn limit(&self) -> Duration {
        let hours = self.entry.run().config.compute.map_or(folo_api::DEFAULT_MAX_HOURS, |c| c.max_hours);
        self.core.tuning.max_run.unwrap_or(Duration::from_secs_f64(hours as f64 * 3600.0))
    }

    fn hours(&self) -> String {
        match self.core.tuning.max_run {
            Some(d) => format!("{:.1} s", d.as_secs_f64()),
            None => format!("{} h", self.entry.run().config.compute.map_or(0.0, |c| c.max_hours)),
        }
    }

    /// Sets the time limit's end, from now.
    fn arm_deadline(&mut self) {
        let deadline = unix_now() + self.limit().as_secs_f64();
        self.private.deadline = Some(deadline);
        self.entry.update_cloud(|c| c.deadline = Some(rfc3339(deadline)));
    }

    fn past_deadline(&self) -> bool {
        self.private.deadline.is_some_and(|d| unix_now() >= d)
    }

    /// Whether the run must stop before its worker trains: the process shuts down, its instance was terminated by
    /// hand, a stop was asked for, or the time limit ended.
    fn interrupted(&self) -> std::result::Result<(), Halt> {
        if self.core.halt.is_on() {
            Err(Halt::LetGo)
        } else if self.instance_ended() {
            Err(self.lost_by_hand())
        } else if self.entry.stop_requested() {
            Err(Halt::Stopped("stopped before the worker trained".into()))
        } else if self.past_deadline() {
            Err(Halt::Stopped(format!("the time limit of {} ended the run before it trained", self.hours())))
        } else {
            Ok(())
        }
    }

    /// Waits `d`; `LetGo` when the process shuts down meanwhile.
    fn wait(&self, d: Duration) -> std::result::Result<(), Halt> {
        if self.core.halt.pause(d) { Err(Halt::LetGo) } else { Ok(()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_and_disks() {
        assert_eq!(size(1_234), "1.2 kB");
        assert_eq!(size(4_800_000_000), "4.8 GB");
        assert_eq!(disk_for(0), 20);
        assert_eq!(disk_for(4_800_000_000), 30);
        assert_eq!(disk_for(u64::MAX / 4), 4096);
    }

    #[test]
    fn gone_is_believed_only_while_it_is_the_only_answer() {
        let after = Duration::from_millis(60);
        let running = InstanceState::Running { url: "http://w".into() };
        let mut gone = GoneSince::default();
        assert!(!gone.believed(&InstanceState::Gone, after), "the first answer starts the count");
        std::thread::sleep(after);
        assert!(gone.believed(&InstanceState::Stopped, after), "gone or stopped, for that long");
        // A provider that did not show a new instance at first, then did: one later `gone` (a blip of its API)
        // starts a new count instead of ending the old one.
        assert!(!gone.believed(&running, after));
        assert!(!gone.believed(&InstanceState::Gone, after));
        assert!(!gone.believed(&InstanceState::Starting, after));
        std::thread::sleep(after);
        assert!(!gone.believed(&InstanceState::Gone, after), "a count that only starts here");
        std::thread::sleep(after);
        assert!(gone.believed(&InstanceState::Gone, after));
    }

    #[test]
    fn a_copy_that_is_dropped_leaves_no_staged_file_and_tells_its_thread() {
        let dir = folo_test_support::TestDir::new("cloud-copying");
        let (staged, newer) = (staged_copy(&dir), staged_copy(&dir));
        assert_ne!(staged, newer, "every copy has a staging file of its own");
        std::fs::write(&staged, b"an older checkpoint").unwrap();
        std::fs::write(&newer, b"a newer copy's download").unwrap();
        let cancel = Arc::new(AtomicBool::new(false));
        drop(Copying { thread: None, cancel: cancel.clone(), staged: staged.clone() });
        assert!(cancel.load(Ordering::SeqCst) && !staged.exists());
        assert!(newer.exists(), "an old copy's clean-up leaves a newer copy's file alone");
        // The names are ones a later run of the executor sweeps away, should both sides miss one.
        std::fs::write(&staged, b"x").unwrap();
        transfer::remove_partial(&dir);
        assert!(!staged.exists() && !newer.exists());
    }
}
