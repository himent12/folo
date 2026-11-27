//! The cloud control plane (docs/CLOUD.md): what turns a run's `TrainConfig.compute` into a run on another
//! machine's GPU, and makes sure nothing rented is left behind.
//!
//! - [`connections`]: `cloud/connections.json`, the connections and their credentials (never served).
//! - [`ledger`]: `cloud/instances/<id>.json`, one record per launched instance, written before the provider is
//!   asked and at every transition.
//! - [`reaper`]: termination (retried until the provider confirms) and the background thread that finishes what
//!   requests and crashes left: terminations, launches of runs nobody follows, old records.
//! - [`executor`]: a cloud run's own thread through the phases of docs/CLOUD.md, and the runs a previous process
//!   left.
//! - [`worker`]: the worker check, and the worker's history as a local run folder's files.
//! - [`state`]: a run's private state (`runs/<id>/cloud.json`); [`files`]: owner-only files and the locks.
//!
//! [`Cloud`] is the entry point: [`crate::RealStore`] serves `/api/cloud` and cloud runs with it, `folo cloud` uses
//! it on a data dir without a server. Providers are `folo_cloud`'s; tests bring their own.

mod connections;
mod executor;
mod files;
mod ledger;
mod reaper;
mod state;
mod worker;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use folo_api::{
    CloudConnection, CloudInstance, CloudOffer, CloudOfferList, CloudProvider, Compute, ConnectCloud,
    MAX_MAX_HOURS, MIN_MAX_HOURS, RunStatus,
};
use folo_cloud::{Provider, Values};

use crate::error::{Error, Result};
use crate::util::{rfc3339, unix_now};
use connections::{Connections, Stored};
use ledger::Ledger;

pub(crate) use connections::{not_found as connection_not_found, redact};
pub(crate) use executor::{Job, Mode};
pub use ledger::KEEP_ENDED;
pub(crate) use ledger::not_found as instance_not_found;

/// How long an offers answer is reused (`?refresh=true` asks again).
pub const OFFERS_TTL: Duration = Duration::from_secs(60);
/// Smallest and largest `compute.disk_gb`.
pub const DISK_GB: std::ops::RangeInclusive<u32> = 10..=4096;
/// How long [`Cloud::shutdown`] waits for its threads to let go of their runs.
pub const SHUTDOWN_WAIT: Duration = Duration::from_secs(30);

/// The timings of the control plane (docs/CLOUD.md); tests shorten them.
#[derive(Clone, Debug)]
pub struct Tuning {
    /// Between two polls of a worker (or of a booting instance).
    pub poll: Duration,
    /// The longest wait between polls of a worker that does not answer (the wait doubles up to it).
    pub poll_max: Duration,
    /// How long a provisioned instance has to pass the worker check after its launch.
    pub boot_timeout: Duration,
    /// How long a server that already runs (`remote`) has to answer.
    pub remote_boot_timeout: Duration,
    /// `weights/last.folo` is copied after an epoch when the last copy is older than this.
    pub checkpoint_every: Duration,
    /// Between two passes of the reaper.
    pub reaper_every: Duration,
    /// Between two looks for an instance that was never found, after the first, until its watchdog is due
    /// (docs/CLOUD.md, *Instances are never left behind*).
    pub unseen_every: Duration,
    /// Between two looks for an instance that was never found, once its watchdog is due.
    pub unseen_late_every: Duration,
    /// How long a worker may stay unreachable while its provider says the instance is gone (or a `remote` server
    /// stays unreachable) before the run fails with `cloud_instance_lost`; also how long a stopped worker run may
    /// take to end.
    pub lost_grace: Duration,
    /// How long a run waits for its instance's termination to be confirmed before it finishes with a warning (the
    /// reaper carries on).
    pub release_wait: Duration,
    /// A provisioned worker must train on a GPU (docs/CLOUD.md, worker check).
    pub require_gpu: bool,
    /// Every run's time limit, instead of its `compute.max_hours` (test support: a deadline in seconds).
    pub max_run: Option<Duration>,
    /// Maximum automatic re-provision + checkpoint-resume attempts when a provisioned worker is preempted or
    /// unexpectedly disappears mid-training after at least one checkpoint was synced (`0` disables auto-resume).
    pub max_preemption_retries: u32,
    /// Wait before provisioning a replacement worker after preemption (honors stop requests and shutdown
    /// immediately).
    pub preemption_backoff: Duration,
}

impl Default for Tuning {
    fn default() -> Tuning {
        Tuning {
            poll: Duration::from_secs(1),
            poll_max: Duration::from_secs(30),
            boot_timeout: Duration::from_secs(15 * 60),
            remote_boot_timeout: Duration::from_secs(60),
            checkpoint_every: Duration::from_secs(10 * 60),
            reaper_every: Duration::from_secs(30),
            unseen_every: Duration::from_secs(10 * 60),
            unseen_late_every: Duration::from_secs(60 * 60),
            lost_grace: Duration::from_secs(10 * 60),
            release_wait: Duration::from_secs(60),
            require_gpu: true,
            max_run: None,
            max_preemption_retries: 3,
            preemption_backoff: Duration::from_secs(10),
        }
    }
}

/// A provider error as an API error: [`folo_cloud::Error::code`] with its status from `folo_api::codes`, and a hint
/// for a provider that `provisions` instances (a console, an API), or for a server that already runs (`remote`: its
/// token, whether it runs and can be reached).
pub(crate) fn provider_error(provisions: bool) -> impl Fn(folo_cloud::Error) -> Error {
    move |e| provider_error_of(e, provisions)
}

fn provider_error_of(e: folo_cloud::Error, provisions: bool) -> Error {
    let code = e.code();
    let status = folo_api::codes::find(code)
        .and_then(|c| StatusCode::from_u16(c.status).ok())
        .unwrap_or(StatusCode::BAD_GATEWAY);
    let hint = match e.kind {
        folo_cloud::ErrorKind::Auth if provisions => {
            Some("check the credentials in the provider's console, then connect again")
        }
        folo_cloud::ErrorKind::Auth => {
            Some("check the server's `--token` (or FOLO_TOKEN), then connect again")
        }
        folo_cloud::ErrorKind::Unavailable if provisions => {
            Some("the provider's API did not answer as expected; try again")
        }
        folo_cloud::ErrorKind::Unavailable => {
            Some("check that the server runs and that this machine can reach its address, then try again")
        }
        folo_cloud::ErrorKind::NoOffer => {
            Some("GET /api/cloud/connections/{id}/offers lists what the connection can run on now")
        }
        _ => None,
    };
    let err = Error::new(status, code, e.message);
    match hint {
        Some(h) => err.hint(h),
        None => err,
    }
}

/// Tells the threads of a [`Core`] that the process shuts down, and wakes them from their waits.
#[derive(Default)]
pub(crate) struct Halt {
    on: Mutex<bool>,
    wake: Condvar,
}

impl Halt {
    /// The process shuts down.
    pub fn is_on(&self) -> bool {
        *self.on.lock().unwrap()
    }

    /// Waits `d` unless the process shuts down first; true when it does.
    pub fn pause(&self, d: Duration) -> bool {
        let on = self.on.lock().unwrap();
        *self.wake.wait_timeout_while(on, d, |on| !*on).unwrap().0
    }

    fn set(&self) {
        *self.on.lock().unwrap() = true;
        self.wake.notify_all();
    }
}

/// What every part of the control plane shares.
pub(crate) struct Core {
    data_dir: PathBuf,
    providers: Vec<Arc<dyn Provider>>,
    pub tuning: Tuning,
    connections: Connections,
    pub ledger: Ledger,
    offers: Mutex<HashMap<String, (Instant, CloudOfferList)>>,
    pub halt: Halt,
}

/// A connection with its provider.
pub(crate) struct Connected {
    pub connection: Stored,
    pub provider: Arc<dyn Provider>,
}

impl Core {
    fn provider(&self, id: &str) -> Result<Arc<dyn Provider>> {
        self.providers.iter().find(|p| p.info().id == id).cloned().ok_or_else(|| {
            Error::not_found("provider", id).hint(
                "GET /api/cloud/providers (`folo cloud providers`) lists the providers this server knows",
            )
        })
    }

    /// Connection `id` with its provider.
    pub fn connected(&self, id: &str) -> Result<Connected> {
        let connection = self.connections.get(id)?;
        let provider = self.provider(&connection.provider)?;
        Ok(Connected { connection, provider })
    }

    /// The run folder of run `id`.
    pub fn run_dir(&self, id: &str) -> PathBuf {
        self.data_dir.join("runs").join(id)
    }

    /// What connection `c` can run on, from the cache unless it is older than [`OFFERS_TTL`] or `refresh`.
    fn offers(&self, c: &Connected, refresh: bool) -> Result<CloudOfferList> {
        if !refresh
            && let Some((at, list)) = self.offers.lock().unwrap().get(&c.connection.id)
            && at.elapsed() < OFFERS_TTL
        {
            return Ok(list.clone());
        }
        let offers = c
            .provider
            .offers(&c.connection.values())
            .map_err(provider_error(c.provider.info().provisions))?;
        let list = CloudOfferList { offers, fetched: rfc3339(unix_now()) };
        self.offers.lock().unwrap().insert(c.connection.id.clone(), (Instant::now(), list.clone()));
        Ok(list)
    }

    /// The `409 connection_in_use` of connection `id`, when one of its instances is alive or one of its runs active
    /// (from the run folders, so a run of another process counts too). Called while `Connections::remove` holds
    /// `cloud/.lock`, so the ledger is read without taking that lock again to prune.
    fn in_use(&self, id: &str) -> Result<()> {
        let live = self
            .ledger
            .list_without_pruning()
            .into_iter()
            .filter(|i| i.connection_id == id && i.status.is_alive())
            .count();
        let runs = active_runs(&self.data_dir.join("runs"), id);
        if live == 0 && runs.is_empty() {
            return Ok(());
        }
        let mut what = Vec::new();
        if live > 0 {
            what.push(format!("{live} live instance(s)"));
        }
        if !runs.is_empty() {
            what.push(format!("active run(s) {}", runs.join(", ")));
        }
        Err(Error::conflict("connection_in_use", format!("connection '{id}' has {}", what.join(" and ")))
            .hint("stop its runs and wait until GET /api/cloud/instances shows its instances terminated"))
    }
}

/// The active runs under `runs_dir` (their `run.json`) that compute on connection `id`.
fn active_runs(runs_dir: &Path, id: &str) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(runs_dir) else { return Vec::new() };
    let mut out = Vec::new();
    for dir in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
        let Some(file) = std::fs::read(dir.join("run.json")).ok() else { continue };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&file) else { continue };
        let run = &v["run"];
        let active = serde_json::from_value::<RunStatus>(run["status"].clone()).is_ok_and(|s| s.is_active());
        if active && run["config"]["compute"]["connection_id"] == id {
            out.push(run["id"].as_str().unwrap_or_default().to_string());
        }
    }
    out.sort();
    out
}

/// The control plane of one data dir. See the module docs.
pub struct Cloud {
    core: Arc<Core>,
    /// The reaper and the runs' executors, joined at [`Cloud::shutdown`].
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl Cloud {
    /// The control plane of `data_dir` with FOLO's providers ([`folo_cloud::providers`]) and the default timings.
    /// Nothing runs in the background until [`crate::RealStore`] starts it; `folo cloud` uses this as it is.
    pub fn open(data_dir: &Path) -> Cloud {
        Cloud::with(data_dir, folo_cloud::providers(), Tuning::default())
    }

    /// [`Cloud::open`] with these providers and timings.
    pub fn with(data_dir: &Path, providers: Vec<Arc<dyn Provider>>, tuning: Tuning) -> Cloud {
        let dir = data_dir.join("cloud");
        let lock = dir.join(".lock");
        let self_terminating =
            providers.iter().map(|p| p.info()).filter(|p| p.self_terminates).map(|p| p.id).collect();
        let ledger = Ledger::new(&dir, lock.clone(), self_terminating);
        let core = Core {
            data_dir: data_dir.to_path_buf(),
            providers,
            tuning,
            connections: Connections::new(&dir, lock),
            ledger,
            offers: Mutex::default(),
            halt: Halt::default(),
        };
        Cloud { core: Arc::new(core), threads: Mutex::default() }
    }

    /// The providers, in display order (`GET /api/cloud/providers`).
    pub fn providers(&self) -> Vec<CloudProvider> {
        self.core.providers.iter().map(|p| p.info()).collect()
    }

    /// Every connection, oldest first, with what its last check found (`GET /api/cloud/connections`).
    pub fn connections(&self) -> Result<Vec<CloudConnection>> {
        let stored = self.core.connections.list()?;
        Ok(stored.iter().map(|c| self.api(c)).collect())
    }

    fn api(&self, c: &Stored) -> CloudConnection {
        let info = self.core.provider(&c.provider).ok().map(|p| p.info());
        c.api(info.as_ref())
    }

    /// Makes a connection: its values are checked with the provider before anything is stored (`POST
    /// /api/cloud/connections`).
    pub fn connect(&self, req: ConnectCloud) -> Result<CloudConnection> {
        let provider = self.core.provider(&req.provider)?;
        let info = provider.info();
        if let Some(unknown) = req.values.keys().find(|k| !info.fields.iter().any(|f| &f.key == *k)) {
            let fields: Vec<&str> = info.fields.iter().map(|f| f.key.as_str()).collect();
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_body",
                format!("{} has no field `{unknown}`", info.name),
            )
            .hint(format!("its fields are {}", fields.join(", "))));
        }
        let values = Values::from(req.values);
        let account = provider.check(&values).map_err(provider_error(info.provisions))?;
        let stored = self.core.connections.add(&info, &req.name, values, account)?;
        tracing::info!(connection = %stored.id, provider = %stored.provider, "cloud connection made");
        Ok(stored.api(Some(&info)))
    }

    /// Checks connection `id`'s stored values again and records the result; a failed check is the answer's
    /// `status: error`, not an error (`POST /api/cloud/connections/{id}/check`).
    pub fn check(&self, id: &str) -> Result<CloudConnection> {
        let stored = self.core.connections.get(id)?;
        let result = match self.core.provider(&stored.provider) {
            Ok(p) => p.check(&stored.values()),
            Err(e) => Err(folo_cloud::Error::new(folo_cloud::ErrorKind::Invalid, e.body.message)),
        };
        let info = self.core.provider(&stored.provider).ok().map(|p| p.info());
        let stored = self.core.connections.update(id, |c| c.checked(result, info.as_ref()))?;
        Ok(self.api(&stored))
    }

    /// Forgets connection `id`, unless it has live instances or active runs (`DELETE /api/cloud/connections/{id}`).
    pub fn disconnect(&self, id: &str) -> Result<()> {
        self.core.connections.remove(id, || self.core.in_use(id))?;
        self.core.offers.lock().unwrap().remove(id);
        Ok(())
    }

    /// What connection `id` can run on now (`GET /api/cloud/connections/{id}/offers`).
    pub fn offers(&self, id: &str, refresh: bool) -> Result<CloudOfferList> {
        let c = self.core.connected(id)?;
        self.core.offers(&c, refresh)
    }

    /// The instances FOLO launched: alive first, then those that ended in the last [`KEEP_ENDED`], newest first
    /// (`GET /api/cloud/instances`).
    pub fn instances(&self) -> Vec<CloudInstance> {
        self.core.ledger.list()
    }

    /// Terminates instance `id` now: one request to the provider, answered with the record (`terminated` when the
    /// provider confirmed, else `terminating` with the error, which the reaper retries). A run still using it fails
    /// with `cloud_instance_lost` (`DELETE /api/cloud/instances/{id}`, `folo cloud terminate`).
    pub fn terminate(&self, id: &str) -> Result<CloudInstance> {
        self.core.terminate_once(id)
    }

    /// Spot preemption auto-resume status of a cloud run (`None` for local runs).
    pub(crate) fn preemption_status(
        &self,
        entry: &crate::runs::RunEntry,
    ) -> Option<crate::store::CloudPreemptionStatus> {
        let run = entry.run();
        if run.cloud.is_none() && run.config.compute.is_none() {
            return None;
        }
        let priv_state = state::Private::load(entry.dir());
        let max_retries = entry.max_preempt_retries().unwrap_or(self.core.tuning.max_preemption_retries);
        Some(crate::store::CloudPreemptionStatus {
            retries: priv_state.preemptions,
            max_retries,
            recovering: priv_state.recovering_preemption && run.status.is_active(),
            preempted_instances: priv_state.prior_instance_ids,
        })
    }

    /// What a run with `compute` would execute on, checked as `POST /api/runs` checks it: the limits, the
    /// connection and the offer (read through the offers cache).
    pub(crate) fn target(&self, compute: &Compute) -> Result<(Connected, CloudOffer)> {
        check_limits(compute)?;
        let c = self.core.connected(&compute.connection_id)?;
        let offers = self.core.offers(&c, false)?;
        let offer = find_offer(offers.offers, compute)?;
        Ok((c, offer))
    }

    /// Starts executing the cloud run of `job` on a thread of its own, holding the run's `cloud.lock`: a run this
    /// process registered (and claimed), or one a previous process left (claimed here). `409 run_active` when
    /// another process follows the run.
    pub(crate) fn start(&self, job: Job, mode: Mode) -> Result<()> {
        let dir = job.entry.dir().to_path_buf();
        let lock = files::try_lock_run(&dir)
            .map_err(|e| Error::io(format!("locking {}: {e}", files::run_lock_path(&dir).display())))?
            .ok_or_else(|| crate::runs::followed_elsewhere(job.entry.id()))?;
        if mode == Mode::Recover && !job.entry.follow() {
            return Ok(());
        }
        let handle = executor::spawn(self.core.clone(), job, lock, mode)
            .map_err(|e| Error::internal(format!("could not start the cloud run's thread: {e}")))?;
        self.keep(handle);
        Ok(())
    }

    fn keep(&self, handle: JoinHandle<()>) {
        let mut threads = self.threads.lock().unwrap();
        threads.retain(|t| !t.is_finished());
        threads.push(handle);
    }

    /// Picks up the active cloud runs a previous process left in `runs` (those no other process follows): a run
    /// that was training is followed again and fetches what it missed; one in an earlier phase is released and
    /// becomes `stopped`. Then starts the reaper, which leaves the instances of followed runs alone.
    pub(crate) fn recover_and_reap(&self, runs: &crate::runs::Runs) {
        for entry in runs.entries() {
            let run = entry.run();
            let Some(compute) = run.config.compute.as_ref().filter(|_| run.status.is_active()) else {
                continue;
            };
            let started = match self.core.connected(&compute.connection_id) {
                Ok(connected) => {
                    self.start(Job { entry: entry.clone(), connected, upload: None }, Mode::Recover)
                }
                Err(e) => executor::abandon(&self.core, &entry, &e.body.message),
            };
            if let Err(e) = started {
                tracing::info!(run = %run.id, "not following the cloud run: {}", e.body.message);
            }
        }
        match reaper::spawn(self.core.clone()) {
            Ok(handle) => self.keep(handle),
            Err(e) => tracing::error!("could not start the cloud reaper: {e}"),
        }
    }

    /// Stops the background work: cloud runs let go of their workers (they keep training, and the next process
    /// follows them again), the reaper ends. Waits at most [`SHUTDOWN_WAIT`] for the threads.
    pub fn shutdown(&self) {
        self.core.halt.set();
        let threads = std::mem::take(&mut *self.threads.lock().unwrap());
        let t0 = Instant::now();
        for t in threads {
            while !t.is_finished() && t0.elapsed() < SHUTDOWN_WAIT {
                std::thread::sleep(Duration::from_millis(20));
            }
            if t.is_finished() {
                let _ = t.join();
            }
        }
    }
}

impl Drop for Cloud {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// The device a cloud run's time is estimated for: one no measured row of this server is keyed by, so the estimate
/// is the reference's ([`cloud_estimate`] names it): the GPU reference, or the CPU's for a `remote` server that
/// offers its CPU.
pub(crate) fn estimate_device(offer: &CloudOffer) -> crate::throughput::DeviceId {
    let device = if offer.id == folo_cloud::remote::CPU_OFFER { "cpu:cloud" } else { "gpu:cloud" };
    crate::throughput::DeviceId { device: device.into(), device_name: offer.gpu.clone() }
}

/// `POST /api/runs/estimate` with `compute`: the reference estimate (for [`estimate_device`]) as the cloud's, with
/// what the run would cost on `offer` (`cost`; absent without a price) and at most within `max_hours`.
pub(crate) fn cloud_estimate(
    mut e: folo_api::RunEstimate,
    offer: &CloudOffer,
    max_hours: f32,
) -> folo_api::RunEstimate {
    e.device = "cloud".into();
    e.device_name = offer.gpu.clone();
    e.basis.note = format!(
        "{}; a cloud run's time is the reference machine's, and the offer's {} may be faster or slower",
        e.basis.note, offer.gpu
    );
    let usd = |seconds: f64, price: f64| (seconds / 3600.0 * price * 100.0).round() / 100.0;
    e.cost = offer.price_per_hour.map(|p| folo_api::CostEstimate {
        price_per_hour: p,
        est_usd: usd(e.est_total_s, p),
        est_usd_low: usd(e.est_total_s_low, p),
        est_usd_high: usd(e.est_total_s_high, p),
        max_usd: usd(max_hours as f64 * 3600.0, p),
    });
    e
}

/// `400 invalid_config` for a `compute.max_hours` or `compute.disk_gb` out of range.
pub(crate) fn check_limits(compute: &Compute) -> Result<()> {
    let bad = |m: String| Error::new(StatusCode::BAD_REQUEST, "invalid_config", m);
    if !(MIN_MAX_HOURS..=MAX_MAX_HOURS).contains(&compute.max_hours) {
        return Err(bad(format!(
            "compute.max_hours must be in {MIN_MAX_HOURS}..={MAX_MAX_HOURS}, got {}",
            compute.max_hours
        )));
    }
    if let Some(gb) = compute.disk_gb.filter(|gb| !DISK_GB.contains(gb)) {
        return Err(bad(format!(
            "compute.disk_gb must be in {}..={}, got {gb}",
            DISK_GB.start(),
            DISK_GB.end()
        )));
    }
    Ok(())
}

/// `compute.offer_id` among `offers`; `404 offer_not_found` when the connection has no such offer (any more).
pub(crate) fn find_offer(offers: Vec<CloudOffer>, compute: &Compute) -> Result<CloudOffer> {
    offers.into_iter().find(|o| o.id == compute.offer_id).ok_or_else(|| {
        Error::not_found("offer", &compute.offer_id).hint(format!(
            "GET /api/cloud/connections/{}/offers lists what the connection can run on now",
            compute.connection_id
        ))
    })
}

/// `400 invalid_config` unless `config.device` is `auto`: a cloud run trains on the device its worker picks.
pub(crate) fn check_device(config: &folo_api::TrainConfig) -> Result<()> {
    if matches!(config.device.trim().to_ascii_lowercase().as_str(), "" | "auto") {
        return Ok(());
    }
    Err(Error::new(
        StatusCode::BAD_REQUEST,
        "invalid_config",
        format!("device is `{}`, but a cloud run trains on the device its worker picks", config.device),
    )
    .hint("leave `device` out (or set it to auto) when `compute` is set"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_provider_error_points_where_its_provider_is_fixed() {
        let hint = |kind, provisions| {
            provider_error(provisions)(folo_cloud::Error::new(kind, "x")).body.hint.unwrap_or_default()
        };
        // RunPod: its console; a remote server: its token, and whether it runs and can be reached.
        assert!(hint(folo_cloud::ErrorKind::Auth, true).contains("provider's console"));
        assert!(hint(folo_cloud::ErrorKind::Auth, false).contains("`--token`"));
        assert!(hint(folo_cloud::ErrorKind::Unavailable, true).contains("provider's API"));
        let remote = hint(folo_cloud::ErrorKind::Unavailable, false);
        assert!(remote.contains("the server runs") && !remote.contains("provider"), "{remote}");
    }

    #[test]
    fn disconnect_does_not_deadlock_when_expired_ended_instance_exists() {
        use folo_api::{CloudAvailability, CloudField, CloudInstanceStatus, CloudOffer, CloudProvider};
        use folo_cloud::{Account, Values};

        struct StubProvider;
        impl Provider for StubProvider {
            fn info(&self) -> CloudProvider {
                CloudProvider {
                    id: "stub".into(),
                    name: "Stub".into(),
                    about: String::new(),
                    docs_url: None,
                    fields: vec![CloudField {
                        key: "api_key".into(),
                        label: "Key".into(),
                        secret: true,
                        required: true,
                        placeholder: None,
                        help: None,
                    }],
                    provisions: true,
                    self_terminates: false,
                }
            }
            fn check(&self, _: &Values) -> folo_cloud::Result<Account> {
                Ok(Account::default())
            }
            fn offers(&self, _: &Values) -> folo_cloud::Result<Vec<CloudOffer>> {
                Ok(Vec::new())
            }
        }

        let dir = folo_test_support::TestDir::new("cloud-disconnect-expired");
        let cloud = Arc::new(Cloud::with(&dir, vec![Arc::new(StubProvider)], Tuning::default()));
        let values = [("api_key".to_string(), "key-0123456789".to_string())].into_iter().collect();
        let conn =
            cloud.connect(ConnectCloud { provider: "stub".into(), name: String::new(), values }).unwrap();
        let offer = CloudOffer {
            id: "o".into(),
            gpu: "G".into(),
            gpu_count: 1,
            vram_bytes: 0,
            price_per_hour: None,
            region: None,
            detail: None,
            availability: CloudAvailability::High,
        };
        let inst = cloud.core.ledger.create(&conn.id, "stub", "run_old", &offer, None, None).unwrap();
        let expired_at = rfc3339(unix_now() - KEEP_ENDED.as_secs_f64() - 60.0);
        cloud
            .core
            .ledger
            .update(&inst.id, |i| {
                i.status = CloudInstanceStatus::Terminated;
                i.ended = Some(expired_at.clone());
            })
            .unwrap();

        let (tx, rx) = std::sync::mpsc::channel();
        let c = cloud.clone();
        let id = conn.id.clone();
        let handle = std::thread::spawn(move || {
            let _ = tx.send(c.disconnect(&id));
        });
        let res = rx
            .recv_timeout(Duration::from_secs(2))
            .expect("disconnect deadlocked on cloud/.lock while pruning an expired ended instance");
        handle.join().unwrap();
        res.unwrap();
        assert!(cloud.connections().unwrap().is_empty());
    }
}
