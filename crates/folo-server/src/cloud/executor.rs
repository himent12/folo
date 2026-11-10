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

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use folo_api::{
    CloudInstanceStatus, CloudOffer, CloudPhase, CreateUpload, DatasetFormat, Health, LogLevel, Run,
    RunEvent, RunList, RunLogs, RunMetrics, RunSamples, RunStatus, StatusEvent, SystemResponse,
    TransferProgress, UploadKind, Uploaded,
};
use folo_client::transfer::{self, Control, TransferError};
use folo_client::{Client, encode};
use folo_cloud::{ErrorKind, InstanceState, LaunchSpec};
use folo_data::Dataset as _;
use folo_data::LoadedDataset;

use super::files::{self, RunLock};
use super::state::{Outcome, Private};
use super::{Connected, Core, worker};
use crate::error::Result;
use crate::runs::{Ended, RunEntry, run_model_id};
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

    // ---- phases ----------------------------------------------------------------------------------------------------

    fn start(&mut self, release: &mut Release) -> std::result::Result<Outcome, Halt> {
        let ds = self.open_dataset()?;
        let endpoint = self.provision(&ds, release)?;
        let client = self.boot(endpoint)?;
        let synced = self.sync(&client, &ds)?;
        drop(ds);
        self.start_worker_run(&client, synced)?;
        self.train(&client)
    }

    fn recover(&mut self) -> std::result::Result<Outcome, Halt> {
        // Collected before the restart: what is left is the release.
        if let Some(o) = self.private.outcome.clone() {
            return Ok(o);
        }
        let phase = self.phase_now();
        let (Some(url), Some(_)) = (self.private.worker_url.clone(), &self.private.run_id) else {
            return Err(Halt::Interrupted(phase));
        };
        if phase < CloudPhase::Training {
            return Err(Halt::Interrupted(phase));
        }
        let token = match self.c.provider.fixed_endpoint(&self.c.connection.values()) {
            Some(e) => e.token,
            None => self.private.token.clone().unwrap_or_default(),
        };
        self.entry.log(
            LogLevel::Info,
            "this server restarted: following the run on its worker again and fetching what it missed",
        );
        self.train(&worker::client(&url, &token))
    }

    /// The local dataset, opened as the local trainer opens it, before anything is rented.
    fn open_dataset(&mut self) -> std::result::Result<LoadedDataset, Halt> {
        let d = self.entry.dataset();
        let opts = folo_data::OpenOptions { format: crate::runs::data_format(d.format), verify_images: true };
        let ds = folo_data::open_with(Path::new(&d.path), &opts)
            .map_err(|e| failed("data_error", format!("opening the dataset {}: {e}", d.path)))?;
        if ds.train.is_empty() {
            return Err(failed("invalid_dataset", "the dataset has no training images"));
        }
        self.private.classes = ds.classes.clone();
        Ok(ds)
    }

    fn provision(
        &mut self,
        ds: &LoadedDataset,
        release: &mut Release,
    ) -> std::result::Result<Endpoint, Halt> {
        let values = self.c.connection.values();
        if let Some(e) = self.c.provider.fixed_endpoint(&values) {
            return Ok(Endpoint::Fixed(e));
        }
        let info = self.c.provider.info();
        let offer = self.offer();
        let compute = self.entry.run().config.compute.expect("a cloud run has compute");
        let len = folo_data::pack::packed_len(ds)
            .map_err(|e| failed("data_error", format!("measuring the dataset: {e}")))?;
        let disk_gb = compute.disk_gb.unwrap_or_else(|| disk_for(len));
        self.entry
            .log(LogLevel::Info, format!("launching {} at {} ({disk_gb} GB disk)", offer.gpu, info.name));
        let internal = |e: std::io::Error| failed("internal", e.to_string());
        let token = files::random_hex(32).map_err(internal)?;
        self.arm_deadline();
        let deadline = self.private.deadline.map(rfc3339);
        let record = self
            .core
            .ledger
            .create(&self.c.connection.id, &info.id, self.entry.id(), &offer, deadline, Some(self.limit()))
            .map_err(|e| failed(&e.body.code, e.body.message))?;
        release.arm(&record.id);
        self.private.instance_id = Some(record.id.clone());
        self.private.token = Some(token.clone());
        self.save();
        self.entry.update_cloud(|c| c.instance_id = Some(record.id.clone()));
        let spec = LaunchSpec {
            name: record.name.clone(),
            offer_id: offer.id.clone(),
            disk_gb,
            token,
            lifetime_s: self.limit().as_secs(),
            worker_base_url: folo_cloud::worker_base_url(&values, env!("CARGO_PKG_VERSION")),
        };
        let nothing = |release: &mut Release, e: folo_cloud::Error| {
            self.core.ledger.remove(&record.id);
            release.disarm();
            self.entry.update_cloud(|c| c.instance_id = None);
            failed(e.code(), e.message)
        };
        let provider_ref = match self.c.provider.launch(&values, &spec) {
            Ok(r) => r,
            // The request may have gone through and only its answer been lost.
            Err(e) if e.kind == ErrorKind::Unavailable => match self.c.provider.find(&values, &spec.name) {
                Ok(refs) if !refs.is_empty() => {
                    self.entry.log(
                        LogLevel::Warn,
                        format!("the launch request failed ({e}), but the instance exists; using it"),
                    );
                    // A name is unique to one launch, so more than one is a provider's retry: those go.
                    for extra in &refs[1..] {
                        if let Err(e) = self.c.provider.terminate(&values, extra) {
                            self.entry
                                .log(LogLevel::Warn, format!("terminating the extra instance {extra}: {e}"));
                        }
                    }
                    refs[0].clone()
                }
                // Not listed (yet) or unknown: the release looks for it by its name again, and the reaper after
                // that (super::reaper::LISTED_WITHIN).
                Ok(_) | Err(_) => return Err(failed(e.code(), e.message)),
            },
            Err(e) => return Err(nothing(release, e)),
        };
        let record = self
            .core
            .ledger
            .update(&record.id, |i| {
                i.provider_ref = Some(provider_ref.clone());
                if i.status == CloudInstanceStatus::Launching {
                    i.status = CloudInstanceStatus::Starting;
                }
            })
            .map_err(|e| failed(&e.body.code, e.body.message))?;
        if !matches!(record.status, CloudInstanceStatus::Starting) {
            // Terminated by hand while the launch was in flight: the instance it created goes too.
            let _ = self.core.ledger.update(&record.id, |i| {
                i.status = CloudInstanceStatus::Terminating;
                i.ended = None;
            });
            return Err(self.lost_by_hand());
        }
        self.interrupted()?;
        Ok(Endpoint::Instance(provider_ref))
    }

    fn boot(&mut self, endpoint: Endpoint) -> std::result::Result<Client, Halt> {
        let provisioned = matches!(endpoint, Endpoint::Instance(_));
        let timeout =
            if provisioned { self.core.tuning.boot_timeout } else { self.core.tuning.remote_boot_timeout };
        let boot_failed = |m: String| failed("cloud_boot_failed", m);
        let minutes = |d: Duration| match d.as_secs() {
            s if s >= 120 => format!("{} minutes", s / 60),
            s => format!("{s} s"),
        };
        let t0 = Instant::now();
        let values = self.c.connection.values();
        // A provider may not show an instance it has only just created: gone (or stopped) is believed once it has
        // been the answer for a while, without another answer in between.
        let mut gone = GoneSince::default();
        let believe_after = self.core.tuning.poll * GONE_POLLS;
        let (url, token, instance) = match endpoint {
            Endpoint::Fixed(e) => {
                self.phase(CloudPhase::Booting, format!("connecting to the FOLO server at {}", e.url));
                self.arm_deadline();
                (e.url, e.token, None)
            }
            Endpoint::Instance(r) => {
                self.phase(CloudPhase::Booting, "waiting for the instance to start");
                let url = loop {
                    self.interrupted()?;
                    match self.c.provider.state(&values, &r) {
                        Ok(state) if gone.believed(&state, believe_after) => {
                            // Terminated by hand meanwhile: lost, not a boot failure.
                            self.interrupted()?;
                            return Err(boot_failed("the instance stopped before it started".into()));
                        }
                        Ok(InstanceState::Running { url }) => break url,
                        Ok(_) => {}
                        Err(e) => tracing::debug!(run = %self.entry.id(), "instance state: {e}"),
                    }
                    if t0.elapsed() >= timeout {
                        return Err(boot_failed(format!(
                            "the instance did not start within {}",
                            minutes(timeout)
                        )));
                    }
                    self.wait(self.core.tuning.poll)?;
                };
                (url, self.private.token.clone().unwrap_or_default(), Some(r))
            }
        };
        if provisioned {
            self.entry.log(
                LogLevel::Info,
                format!(
                    "the instance runs; it installs FOLO (a few minutes), then its worker answers at {url}"
                ),
            );
        }
        let client = worker::client(&url, &token);
        let mut asked = Instant::now();
        let (health, system) = loop {
            self.interrupted()?;
            let answer = client
                .get::<Health>("/health")
                .and_then(|h| Ok((h, client.get::<SystemResponse>("/system")?)));
            match answer {
                Ok(x) => break x,
                Err(e) if e.status.map(|s| s.as_u16()) == Some(401) => {
                    return Err(boot_failed(format!("the worker at {url} rejected the token")));
                }
                Err(e) if t0.elapsed() >= timeout => {
                    return Err(boot_failed(format!(
                        "the worker at {url} did not answer within {} ({e})",
                        minutes(timeout)
                    )));
                }
                Err(_) => {}
            }
            // An instance whose setup fails terminates itself (docs/CLOUD.md, `runpod`): it is gone long before
            // the timeout, and the run says so instead of waiting for a worker that will never answer.
            if let Some(r) =
                instance.as_ref().filter(|_| asked.elapsed() >= self.core.tuning.poll * STATE_POLLS)
            {
                asked = Instant::now();
                match self.c.provider.state(&values, r) {
                    Ok(state) if gone.believed(&state, believe_after) => {
                        self.interrupted()?;
                        return Err(boot_failed(self.setup_failed(&values)));
                    }
                    Ok(_) => {}
                    Err(e) => tracing::debug!(run = %self.entry.id(), "instance state: {e}"),
                }
            }
            self.wait(self.core.tuning.poll)?;
        };
        let amp = self.entry.run().config.amp;
        worker::check(&system, &health.version, provisioned && self.core.tuning.require_gpu, amp)
            .map_err(boot_failed)?;
        if let Some(id) = &self.private.instance_id {
            let _ = self.core.ledger.update(id, |i| {
                if i.status == CloudInstanceStatus::Starting {
                    i.status = CloudInstanceStatus::Running;
                }
            });
            // Its setup has started, so its own clock runs: the latest it can live is known from here.
            let _ = self.core.ledger.booted(id);
        }
        let device = system.capabilities.map_or_else(|| "its default device".to_string(), |c| c.device);
        self.entry
            .log(LogLevel::Info, format!("the worker answers: FOLO {} training on {device}", health.version));
        self.private.worker_url = Some(url);
        self.save();
        Ok(client)
    }

    /// Why an instance that is gone before its worker answered most likely is: its setup failed and it terminated
    /// itself. Names where it installs FOLO from, the step that fails when a release is not published.
    fn setup_failed(&self, values: &folo_cloud::Values) -> String {
        format!(
            "the instance stopped before its worker answered: its setup failed, and it terminated itself. It \
             installs FOLO from {} (the connection's worker download URL names another place); the instance's log \
             at {} says which step failed",
            folo_cloud::worker_base_url(values, env!("CARGO_PKG_VERSION")),
            self.c.provider.info().name
        )
    }

    fn sync(&mut self, client: &Client, ds: &LoadedDataset) -> std::result::Result<Synced, Halt> {
        let d = self.entry.dataset();
        let total = folo_data::pack::packed_len(ds)
            .map_err(|e| failed("data_error", format!("measuring the dataset: {e}")))?;
        let images = ds.train.len() + ds.val.len() + ds.test.len();
        self.phase(
            CloudPhase::Syncing,
            format!("uploading the dataset {} ({images} images, {})", d.name, size(total)),
        );
        let req = CreateUpload {
            kind: UploadKind::Dataset,
            name: d.name.clone(),
            format: DatasetFormat::Yolo,
            fingerprint: self.fingerprint.clone(),
            bytes: Some(total),
        };
        let write = |w: &mut dyn Write| {
            folo_data::pack::pack(ds, w).map(drop).map_err(|e| std::io::Error::other(e.to_string()))
        };
        let (uploaded, sent) = self.upload("the dataset", client, &req, write)?;
        let dataset = uploaded.dataset.ok_or_else(|| {
            failed("cloud_sync_failed", "the worker's answer to the dataset upload has no dataset")
        })?;
        if sent == 0 {
            self.entry.log(LogLevel::Info, "the worker already has this dataset; nothing to send");
        }
        let last = self.dir.join("weights").join("last.folo");
        let start = if last.is_file() {
            Some((last, format!("{}-last", self.entry.run().name), true))
        } else {
            self.entry.weights().map(|w| {
                let name = w.file_stem().map_or("weights".into(), |s| s.to_string_lossy().into_owned());
                (w, name, false)
            })
        };
        let model = match start {
            None => None,
            Some((path, name, checkpoint)) => {
                let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                let what =
                    if checkpoint { "the checkpoint weights/last.folo" } else { "the weights to start from" };
                self.entry.log(LogLevel::Info, format!("uploading {what} ({})", size(bytes)));
                let req = CreateUpload {
                    kind: UploadKind::Model,
                    name,
                    format: DatasetFormat::Auto,
                    fingerprint: file_fingerprint(&path),
                    bytes: Some(bytes),
                };
                let write = |w: &mut dyn Write| {
                    std::fs::File::open(&path).and_then(|mut f| std::io::copy(&mut f, w)).map(drop)
                };
                let (uploaded, _) = self.upload(what, client, &req, write)?;
                let m = uploaded.model.ok_or_else(|| {
                    failed("cloud_sync_failed", "the worker's answer to the weights upload has no model")
                })?;
                Some((m.id, checkpoint))
            }
        };
        self.save();
        Ok(Synced { dataset_id: dataset.id, model })
    }

    /// Sends what `write` produces as upload `req` to the worker, with progress in `Run.cloud`; the bytes sent.
    fn upload(
        &self,
        what: &str,
        client: &Client,
        req: &CreateUpload,
        write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
    ) -> std::result::Result<(Uploaded, u64), Halt> {
        let total = req.bytes.unwrap_or(0);
        let mut sent = 0;
        let (mut shown, mut logged) = (None::<Instant>, Instant::now());
        let entry = &self.entry;
        let mut progress = |p: TransferProgress| {
            sent = p.done_bytes;
            if shown.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY) || p.done_bytes >= total {
                entry.update_cloud(|c| c.progress = Some(p));
                shown = Some(Instant::now());
            }
            if logged.elapsed() >= TRANSFER_LOG_EVERY {
                entry.log(
                    LogLevel::Info,
                    format!("uploading {what}: {} of {}", size(p.done_bytes), size(total)),
                );
                logged = Instant::now();
            }
        };
        let stop = || self.interrupted().is_err();
        let mut ctl = Control { progress: &mut progress, stop: &stop, backoff: transfer::FIRST_BACKOFF };
        let up = transfer::upload(client, req, write, &mut ctl);
        match up {
            Ok(u) => Ok((u, sent)),
            Err(TransferError::Stopped) => Err(self.interrupted().err().unwrap_or(Halt::LetGo)),
            Err(e) => Err(failed("cloud_sync_failed", format!("uploading {what}: {e}"))),
        }
    }

    /// Starts the run on the worker: the run's config without `compute`, on the worker's device and with the ids the
    /// worker gave the dataset and the starting weights. A request whose answer was lost is found again by the run's
    /// name there (`<name> (<run id>)`), so the worker never trains it twice.
    fn start_worker_run(&mut self, client: &Client, synced: Synced) -> std::result::Result<(), Halt> {
        let run = self.entry.run();
        let name = format!("{} ({})", run.name, run.id);
        let mut body = serde_json::to_value(&run.config).expect("configs serialize");
        let fields = body.as_object_mut().expect("a config is an object");
        fields.remove("compute");
        fields.insert("name".into(), name.clone().into());
        fields.insert("device".into(), "auto".into());
        fields.insert("dataset_id".into(), synced.dataset_id.into());
        fields.insert("weights".into(), serde_json::Value::Null);
        match synced.model {
            Some((id, true)) => fields.insert("checkpoint".into(), id.into()),
            Some((id, false)) => fields.insert("weights".into(), id.into()),
            None => None,
        };
        let mut attempt = 1;
        let created = loop {
            match client.post::<_, Run>("/runs", &body) {
                Ok(r) => break r,
                Err(e) if e.is_transient() && attempt < ATTEMPTS => {
                    // Only an active one: a `remote` server keeps the ended runs of the run's earlier attempts,
                    // under the same name.
                    if let Ok(list) = client.get::<RunList>("/runs")
                        && let Some(r) =
                            list.runs.into_iter().find(|r| r.name == name && r.status.is_active())
                    {
                        break r;
                    }
                    attempt += 1;
                    self.wait(self.core.tuning.poll)?;
                }
                Err(e) => {
                    let code = if e.is_transient() { "cloud_sync_failed" } else { e.code() };
                    return Err(failed(code, format!("starting the run on the worker: {e}")));
                }
            }
        };
        self.private.run_id = Some(created.id.clone());
        (self.private.after_step, self.private.after_epoch, self.private.after_log) = (0, 0, 0);
        self.save();
        self.phase(CloudPhase::Training, format!("the worker took the run as {}", created.id));
        Ok(())
    }

    /// Mirrors the worker's run until it ends (stopping it when asked to or at the time limit), then collects.
    fn train(&mut self, client: &Client) -> std::result::Result<Outcome, Halt> {
        let run_id = self.private.run_id.clone().expect("a training cloud run has a worker run");
        let t = self.core.tuning.clone();
        let mut stop_sent: Option<Instant> = None;
        let mut time_up = false;
        let mut wait = t.poll;
        let (mut unreachable, mut gone): (Option<Instant>, Option<Instant>) = (None, None);
        let mut answered = false;
        let mut cost_at = Instant::now();
        let end = loop {
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
    fn instance_ended(&self) -> bool {
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

    fn lost_by_hand(&self) -> Halt {
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

    /// One poll of the worker: its run's status, the steps and epochs after the cursors (each new epoch with its
    /// samples), the new log lines, and the checkpoint when its copy is due. Every step goes to `metrics.jsonl`;
    /// subscribers get every epoch and the newest step. The worker's run once it ended.
    fn mirror(&mut self, client: &Client, run_id: &str) -> folo_client::Result<Option<Run>> {
        let id = encode(run_id);
        let mut started = Instant::now();
        let local = self.entry.id().to_string();
        let mut step = |what: &str| {
            let took = std::mem::replace(&mut started, Instant::now()).elapsed();
            if took >= SLOW_STEP {
                tracing::info!(run = %local, "mirroring the worker: {what} took {:.1} s", took.as_secs_f64());
            }
        };
        let run: Run = client.get(&format!("/runs/{id}"))?;
        step("its run");
        self.follow_status(&run);
        // Always with the cursors (0 before the first step): only then are the steps gap-free.
        let (after_step, after_epoch) = (self.private.after_step, self.private.after_epoch);
        let path = format!("/runs/{id}/metrics?after_step={after_step}&after_epoch={after_epoch}");
        let history: RunMetrics = client.get(&path)?;
        step("its new steps and epochs");
        let newest = history.steps.last().map(|s| s.step);
        let mut epoch_done = false;
        for ev in worker::in_order(history) {
            if let RunEvent::Epoch(e) = &ev {
                self.copy_samples(client, &id, e.epoch)?;
            }
            crate::runs::append_line(
                &self.entry.metrics_path(),
                &folo_vision::train::event_record(&ev).to_string(),
            );
            match &ev {
                RunEvent::Step(s) => {
                    self.private.after_step = s.step;
                    if Some(s.step) == newest {
                        self.entry.on_event(&ev);
                    }
                }
                RunEvent::Epoch(e) => {
                    self.private.after_epoch = e.epoch;
                    self.entry.on_event(&ev);
                    epoch_done = true;
                }
                _ => {}
            }
        }
        step("the new epochs' samples");
        loop {
            let logs: RunLogs =
                client.get(&format!("/runs/{id}/logs?after={}&limit={LOG_PAGE}", self.private.after_log))?;
            let n = logs.lines.len();
            for line in logs.lines {
                self.entry.on_event(&RunEvent::Log(line));
            }
            self.private.after_log += n as u64;
            if n < LOG_PAGE {
                break;
            }
        }
        self.save();
        step("its new log lines");
        self.reap_copy();
        let every = self.core.tuning.checkpoint_every;
        let due = self.private.synced.is_none_or(|t| unix_now() - t >= every.as_secs_f64())
            && self.copy_failed.is_none_or(|t| t.elapsed() >= COPY_RETRY_EVERY.min(every));
        if epoch_done && run.status.is_active() && due && self.copying.is_none() {
            self.start_copy(client, run_id);
        }
        Ok((!run.status.is_active()).then_some(run))
    }

    /// Starts a copy of the worker's `weights/last.folo` on a thread of its own ([`Copying`]): it fetches the file
    /// into a staging file, and [`Exec::reap_copy`] takes its result. It ends early when the process shuts down
    /// or a stop is asked for (the collection at the run's end copies the final file).
    fn start_copy(&mut self, client: &Client, run_id: &str) {
        let weights = self.dir.join("weights");
        let staged = staged_copy(&weights);
        let path = format!("/models/{}/weights", encode(&run_model_id(run_id, true)));
        let cancel = Arc::new(AtomicBool::new(false));
        let (client, core, entry) = (client.clone(), self.core.clone(), self.entry.clone());
        let (cancelled, dest) = (cancel.clone(), staged.clone());
        let copy = move || {
            std::fs::create_dir_all(&weights).map_err(TransferError::Io)?;
            let stop = || cancelled.load(Ordering::SeqCst) || core.halt.is_on() || entry.stop_requested();
            let mut ctl = Control { progress: &mut |_| {}, stop: &stop, backoff: transfer::FIRST_BACKOFF };
            let fetched = transfer::download(&client, &path, &dest, &mut ctl).map(drop);
            // Nobody takes the file any more (see `Copying`'s drop for the order).
            if cancelled.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(&dest);
            }
            fetched
        };
        let name = format!("folo-cloud-copy-{}", self.entry.id());
        match std::thread::Builder::new().name(name).spawn(copy) {
            Ok(thread) => self.copying = Some(Copying { thread: Some(thread), cancel, staged }),
            Err(e) => {
                self.copy_failed = Some(Instant::now());
                tracing::warn!(run = %self.entry.id(), "could not start the checkpoint copy's thread: {e}");
            }
        }
    }

    /// Takes the result of the copy under way once it has ended: the staged file becomes `weights/last.folo`, with
    /// `Run.cloud.synced` and a log line, or a warning (and a later retry) says why not. A worker without the file
    /// yet, and a copy that was told to end, say nothing.
    fn reap_copy(&mut self) {
        if !self.copying.as_ref().is_some_and(|c| c.thread.as_ref().is_none_or(|t| t.is_finished())) {
            return;
        }
        let Some(mut copying) = self.copying.take() else { return };
        let fetched = match copying.thread.take().map(|t| t.join()) {
            Some(Ok(fetched)) => fetched,
            Some(Err(_)) => {
                self.copy_failed = Some(Instant::now());
                tracing::error!(run = %self.entry.id(), "the checkpoint copy's thread panicked");
                return;
            }
            None => return,
        };
        // This thread, which holds the run, is the only one that writes `weights/last.folo`.
        let placed = fetched.and_then(|()| {
            std::fs::rename(&copying.staged, self.dir.join("weights").join("last.folo"))
                .map_err(TransferError::Io)
        });
        match placed {
            Ok(()) => {
                self.copy_failed = None;
                self.synced_now();
            }
            Err(TransferError::Stopped) => {}
            Err(TransferError::Failed(e)) if e.code() == "model_not_found" => {}
            Err(e) => {
                self.copy_failed = Some(Instant::now());
                self.entry.log(LogLevel::Warn, format!("copying weights/last.folo (retried later): {e}"));
            }
        }
    }

    /// `weights/last.folo` in the run folder is the worker's of now.
    fn synced_now(&mut self) {
        let now = unix_now();
        self.private.synced = Some(now);
        self.save();
        self.entry.update_cloud(|c| c.synced = Some(rfc3339(now)));
        self.entry.log(LogLevel::Info, "copied weights/last.folo");
    }

    /// The worker run's status, device and length, on the local run: `queued` while it waits in the worker's queue
    /// (logged when its place changes), then `running`.
    fn follow_status(&mut self, w: &Run) {
        if w.status == RunStatus::Queued {
            if w.queue_position != self.queued_at
                && let Some(p) = w.queue_position
            {
                self.entry
                    .log(LogLevel::Info, format!("waiting in the worker's training queue (position {p})"));
            }
            self.queued_at = w.queue_position;
            return;
        }
        let local = self.entry.run();
        let spe = if w.steps_per_epoch > 0 { w.steps_per_epoch } else { local.steps_per_epoch };
        if (local.device != w.device || local.steps_per_epoch != spe || local.resolved != w.resolved)
            && w.resolved.is_some()
        {
            self.entry.update(|r| {
                r.device = w.device.clone();
                r.steps_per_epoch = spe;
                r.resolved = w.resolved.clone();
            });
        }
        if local.status == RunStatus::Queued {
            let running = StatusEvent::new(RunStatus::Running, None).with_length(w.epochs, spe);
            self.entry.on_event(&RunEvent::Status(running));
        }
    }

    /// The worker's samples of `epoch`, as the local `samples/epoch_NNN.json`.
    fn copy_samples(&self, client: &Client, id: &str, epoch: u32) -> folo_client::Result<()> {
        let samples = match client.get::<RunSamples>(&format!("/runs/{id}/samples?epoch={epoch}")) {
            Ok(s) => s,
            Err(e) if e.code() == "epoch_not_available" => return Ok(()),
            Err(e) => return Err(e),
        };
        let file = worker::samples_file(&samples, &self.private.classes, self.entry.val_offset());
        let path = self.dir.join("samples").join(format!("epoch_{epoch:03}.json"));
        if let Err(e) = crate::index::write_atomic(&path, file.to_string().as_bytes()) {
            tracing::warn!("writing {}: {e}", path.display());
        }
        Ok(())
    }

    /// Copies the worker's `last.folo` (or `best.folo`) into the run folder at the run's end, with
    /// `Run.cloud.progress` following it; false when the worker has none.
    fn copy(
        &mut self,
        client: &Client,
        run_id: &str,
        last: bool,
    ) -> std::result::Result<bool, TransferError> {
        let file = if last { "last.folo" } else { "best.folo" };
        let weights = self.dir.join("weights");
        std::fs::create_dir_all(&weights).map_err(TransferError::Io)?;
        let path = format!("/models/{}/weights", encode(&run_model_id(run_id, last)));
        let entry = &self.entry;
        let mut shown = None::<Instant>;
        let mut progress = |p: TransferProgress| {
            if shown.is_none_or(|t| t.elapsed() >= PROGRESS_EVERY) || p.done_bytes >= p.total_bytes {
                entry.update_cloud(|c| c.progress = Some(p));
                shown = Some(Instant::now());
            }
        };
        let stop = || self.core.halt.is_on();
        let mut ctl = Control { progress: &mut progress, stop: &stop, backoff: transfer::FIRST_BACKOFF };
        match transfer::download(client, &path, &weights.join(file), &mut ctl) {
            Ok(_) => {}
            Err(TransferError::Failed(e)) if e.code() == "model_not_found" => return Ok(false),
            Err(e) => return Err(e),
        }
        if last {
            self.synced_now();
        } else {
            self.entry.log(LogLevel::Info, format!("copied weights/{file}"));
        }
        Ok(true)
    }

    /// The run ended on the worker (`end`; `None` when a stopped run did not end in time): copies both weights files
    /// and returns how it ended. A resumed run's worker writes `best.folo` only once an epoch beats the checkpoint's
    /// best, so the local `best.folo` stays when the worker has none.
    fn collect(
        &mut self,
        client: &Client,
        run_id: &str,
        end: Option<Run>,
        time_up: bool,
    ) -> std::result::Result<Outcome, Halt> {
        // A copy still under way is told to end and not waited for: it only writes its staged file.
        self.copying = None;
        self.phase(CloudPhase::Collecting, "the run ended on the worker; copying its weights");
        for last in [true, false] {
            match self.copy(client, run_id, last) {
                Ok(_) => {}
                Err(TransferError::Stopped) => return Err(Halt::LetGo),
                Err(e) => {
                    let file = if last { "last.folo" } else { "best.folo" };
                    return Err(failed("cloud_sync_failed", format!("copying weights/{file}: {e}")));
                }
            }
        }
        let outcome = match end {
            Some(r) => Outcome { status: r.status, error: r.error, error_code: r.error_code },
            None => Outcome::plain(RunStatus::Stopped),
        };
        // A run that completed (or failed) just as its time was up ended that way, not by the limit.
        if time_up && outcome.status == RunStatus::Stopped {
            self.entry.log(LogLevel::Info, format!("the time limit of {} ended the run", self.hours()));
        }
        Ok(outcome)
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
