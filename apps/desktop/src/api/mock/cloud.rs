//! The demo cloud: a cloud run's simulated life, the RunPod connection and its offers, and how a
//! run's `compute` is checked.

use crate::api::time;
use crate::api::*;
use crate::synth::hash;

use super::training::Clock;
use super::{MOCK_NOW, MockBackend, STEP_DT};

/// How a cloud run ended early.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Ended {
    /// Stopped through the API: the run keeps its checkpoint.
    Stopped,
    /// Its instance was terminated by hand: the run fails with `cloud_instance_lost`.
    InstanceLost,
}

/// A cloud run's life (docs/CLOUD.md, *A cloud run, step by step*) as a function of the clock:
/// provisioning, booting and uploading, then the training curve on the instance, then collecting
/// the weights and releasing the instance. Everything is measured in seconds since the instance
/// was asked for.
#[derive(Clone)]
pub(super) struct CloudSim {
    /// `Run.cloud` as the run started: connection, offer, instance and time limit.
    pub(super) base: RunCloud,
    clock: Clock,
    /// When the instance was asked for (the run's creation), Unix seconds.
    pub(super) t0: f64,
    /// How long provisioning, booting and uploading take.
    pub(super) pre: [f64; 3],
    /// The dataset's upload, bytes.
    pub(super) upload_bytes: u64,
    /// Seconds per training step.
    pub(super) step_dt: f64,
    /// Where the training clock is anchored: (seconds since `t0`, steps done then).
    pub(super) anchor: (f64, u64),
    /// The run's length in steps.
    total: u64,
    /// Ended early: when, and how.
    pub(super) ended: Option<(f64, Ended)>,
    /// Why the provider hasn't confirmed the termination (the instance stays `terminating`).
    pub(super) release_error: Option<String>,
}

/// Seconds of collecting the weights, then of releasing the instance.
pub(super) const COLLECT_S: f64 = 1.;
pub(super) const RELEASE_S: f64 = 0.5;

impl CloudSim {
    /// A run launched as `base` says at `t0` (Unix seconds) that trains `total` steps, at the
    /// demo's pace: about 2 s for each phase before training, then as fast as a local mock run.
    pub(super) fn new(base: RunCloud, clock: Clock, t0: f64, total: u64) -> Self {
        let pre = [2., 2., 2.];
        Self {
            base,
            clock,
            t0,
            pre,
            upload_bytes: 600 << 20,
            step_dt: STEP_DT,
            anchor: (pre.iter().sum(), 0),
            total,
            ended: None,
            release_error: None,
        }
    }

    pub(super) fn elapsed(&self) -> f64 {
        (self.clock.now() - self.t0).max(0.)
    }

    pub(super) fn train_at(&self) -> f64 {
        self.pre.iter().sum()
    }

    /// Training steps done `t` seconds in.
    fn steps_at(&self, t: f64) -> u64 {
        let t = self.ended.map_or(t, |(e, _)| t.min(e));
        if t < self.train_at() {
            return 0;
        }
        (self.anchor.1 + ((t - self.anchor.0).max(0.) / self.step_dt) as u64).min(self.total)
    }

    pub(super) fn steps(&self) -> u64 {
        self.steps_at(self.elapsed())
    }

    /// When training ended (all steps done, or stopped), if it has.
    fn trained_at(&self) -> Option<f64> {
        let natural = self.anchor.0 + self.total.saturating_sub(self.anchor.1) as f64 * self.step_dt;
        let end = self.ended.map_or(natural, |(e, _)| e.min(natural)).max(self.train_at());
        (self.elapsed() >= end).then_some(end)
    }

    /// When the instance was released: right away for a run that ended before training or lost its
    /// instance, else after collecting the weights.
    pub(super) fn released_at(&self) -> Option<f64> {
        match self.ended {
            Some((e, how)) if how == Ended::InstanceLost || e < self.train_at() => Some(e),
            _ => self.trained_at().map(|t| t + COLLECT_S + RELEASE_S).filter(|r| self.elapsed() >= *r),
        }
    }

    fn phase(&self) -> (CloudPhase, Option<TransferProgress>) {
        let t = self.elapsed();
        let [provision, boot, sync] = self.pre;
        if self.released_at().is_some() {
            return (CloudPhase::Released, None);
        }
        if let Some(done) = self.trained_at() {
            return if t < done + COLLECT_S {
                // The two weights files, about 40 MB each for these models.
                let total = 80 << 20;
                let done_bytes = (total as f64 * (t - done) / COLLECT_S) as u64;
                (CloudPhase::Collecting, Some(TransferProgress { done_bytes, total_bytes: total }))
            } else {
                (CloudPhase::Releasing, None)
            };
        }
        if t < provision {
            (CloudPhase::Provisioning, None)
        } else if t < provision + boot {
            (CloudPhase::Booting, None)
        } else if t < self.train_at() {
            let done_bytes = (self.upload_bytes as f64 * (t - provision - boot) / sync) as u64;
            (CloudPhase::Syncing, Some(TransferProgress { done_bytes, total_bytes: self.upload_bytes }))
        } else {
            (CloudPhase::Training, None)
        }
    }

    /// Whether the provider hasn't confirmed the termination: the instance stays `terminating` (and
    /// billed) after the run was released.
    fn stuck(&self) -> bool {
        self.release_error.is_some() && self.trained_at().is_some_and(|t| self.elapsed() >= t + COLLECT_S)
    }

    /// Seconds the instance has existed: until it was released, or until now while it isn't.
    fn lifetime(&self) -> f64 {
        if self.stuck() { self.elapsed() } else { self.released_at().unwrap_or(self.elapsed()) }
    }

    /// Price times `secs`, without a price none.
    fn cost(&self, secs: f64) -> Option<f64> {
        self.base.offer.price_per_hour.map(|p| p * secs / 3600.)
    }

    fn at(&self, secs: f64) -> String {
        time::format((self.t0 + secs) as i64)
    }

    /// `Run.cloud` now.
    fn state(&self) -> RunCloud {
        let (phase, progress) = self.phase();
        let t = self.elapsed().min(self.ended.map_or(f64::MAX, |e| e.0));
        let train_at = self.train_at();
        // `last.folo` is copied every 10 minutes of training, and both files at the end.
        let synced = match self.trained_at() {
            Some(done) if phase >= CloudPhase::Releasing => Some(done + COLLECT_S),
            _ if t >= train_at + 600. => Some(train_at + ((t - train_at) / 600.).floor() * 600.),
            _ => None,
        };
        RunCloud {
            phase,
            progress,
            deadline: (self.elapsed() >= self.pre[0]).then(|| self.at(self.base.max_hours as f64 * 3600.)),
            // Final once released, even while the provider hasn't confirmed the termination.
            cost_usd: self.cost(self.released_at().unwrap_or(self.elapsed())),
            synced: synced.map(|s| self.at(s)),
            ..self.base.clone()
        }
    }

    /// The run as its life says now: status, times, device and `cloud`.
    pub(super) fn apply(&self, run: &mut Run) {
        let cloud = self.state();
        let lost = self.ended.is_some_and(|(_, how)| how == Ended::InstanceLost);
        let stopped = self.ended.is_some_and(|(_, how)| how == Ended::Stopped);
        run.status = match cloud.phase {
            _ if lost => RunStatus::Failed,
            CloudPhase::Released if stopped => RunStatus::Stopped,
            CloudPhase::Released => RunStatus::Completed,
            _ if stopped => RunStatus::Stopping,
            p if p >= CloudPhase::Training => RunStatus::Running,
            _ => RunStatus::Queued,
        };
        let training =
            self.elapsed() >= self.train_at() && !self.ended.is_some_and(|(e, _)| e < self.train_at());
        run.started = training.then(|| self.at(self.train_at()));
        run.finished = self.released_at().map(|r| self.at(r));
        // The worker's GPU once it trains (the offer's catalogue name, as wgpu reports it).
        run.device = if training {
            format!("{} (vulkan)", self.base.offer.id.split('@').next().unwrap_or(&self.base.offer.gpu))
        } else {
            String::new()
        };
        if lost {
            run.error = Some(format!(
                "The instance {} was terminated by hand; the checkpoints copied so far are kept.",
                self.base.instance_id.as_deref().unwrap_or("of this run")
            ));
            run.error_code = Some("cloud_instance_lost".into());
        }
        if !run.status.is_active() {
            run.eta_s = None;
        }
        run.cloud = Some(cloud);
    }

    /// The run's instance as `/api/cloud/instances` lists it.
    pub(super) fn instance(&self, run_id: &str) -> Option<CloudInstance> {
        let id = self.base.instance_id.clone()?;
        let (phase, _) = self.phase();
        let stuck = self.stuck();
        let t = self.elapsed();
        let status = match phase {
            _ if stuck => CloudInstanceStatus::Terminating,
            CloudPhase::Released => CloudInstanceStatus::Terminated,
            CloudPhase::Releasing => CloudInstanceStatus::Terminating,
            CloudPhase::Provisioning => CloudInstanceStatus::Launching,
            CloudPhase::Booting => CloudInstanceStatus::Starting,
            _ => CloudInstanceStatus::Running,
        };
        Some(CloudInstance {
            id,
            connection_id: self.base.connection_id.clone(),
            provider: self.base.provider.clone(),
            provider_ref: (t >= self.pre[0])
                .then(|| format!("{:016x}", hash(&format!("pod{run_id}")))[..14].to_string()),
            name: format!("folo-{run_id}-{:04x}", hash(run_id) & 0xFFFF),
            run_id: Some(run_id.to_string()),
            offer: self.base.offer.clone(),
            status,
            created: self.at(0.),
            ended: (status == CloudInstanceStatus::Terminated).then(|| self.at(self.lifetime())),
            deadline: (t >= self.pre[0]).then(|| self.at(self.base.max_hours as f64 * 3600.)),
            cost_usd: self.cost(self.lifetime()),
            error: self.release_error.clone().filter(|_| stuck),
        })
    }
}

/// `Run.cloud` as a run is launched on `offer`: phase and costs follow from its [`CloudSim`]. A
/// provider that provisions gets an instance per run.
pub(super) fn launch_state(
    run_id: &str,
    compute: &Compute,
    provider: &str,
    offer: CloudOffer,
    provisions: bool,
) -> RunCloud {
    RunCloud {
        connection_id: compute.connection_id.clone(),
        provider: provider.into(),
        offer,
        instance_id: provisions.then(|| format!("ci_{:04x}", hash(run_id) & 0xFFFF)),
        phase: CloudPhase::Provisioning,
        progress: None,
        max_hours: compute.max_hours,
        deadline: None,
        cost_usd: None,
        synced: None,
    }
}

/// The demo's one RunPod connection.
pub(super) const RUNPOD_CONNECTION: &str = "cc_4d2a";
pub(super) const RTX4090_OFFER: &str = "NVIDIA GeForce RTX 4090@secure";
pub(super) const L40S_OFFER: &str = "NVIDIA L40S@secure";
pub(super) const A100_OFFER: &str = "NVIDIA A100 80GB PCIe@secure";

/// The providers as `folo serve` lists them: `info()` of `folo-cloud`'s RunPod and remote server,
/// word for word (the Connect dialog is laid out for these texts).
pub(super) fn cloud_providers() -> Vec<CloudProvider> {
    let field =
        |key: &str, label: &str, secret: bool, required: bool, placeholder: Option<&str>, help: &str| {
            CloudField {
                key: key.into(),
                label: label.into(),
                secret,
                required,
                placeholder: placeholder.map(Into::into),
                help: Some(help.into()),
            }
        };
    vec![
        CloudProvider {
            id: "runpod".into(),
            name: "RunPod".into(),
            about: "Rents an on-demand GPU pod on RunPod for each run and terminates it when the run ends.".into(),
            docs_url: Some("https://www.runpod.io/console/user/settings".into()),
            fields: vec![
                field(
                    "api_key",
                    "API key",
                    true,
                    true,
                    Some("rpa_…"),
                    "RunPod console → Settings → API Keys; the key needs read and write access to pods.",
                ),
                field(
                    "worker_base_url",
                    "Worker download URL",
                    false,
                    false,
                    Some("https://example.com/folo"),
                    "Where instances download FOLO from (install.sh and the release archive) when not the \
                     public release of this version: a mirror, or your own build.",
                ),
            ],
            provisions: true,
            self_terminates: true,
        },
        CloudProvider {
            id: "remote".into(),
            name: "Remote FOLO server".into(),
            about: "A FOLO server you run yourself (`folo serve --host 0.0.0.0 --token …`), on a workstation or a \
                    VM at any cloud."
                .into(),
            docs_url: None,
            fields: vec![
                field(
                    "url",
                    "Server URL",
                    false,
                    true,
                    Some("https://gpu-box.example.com:7878"),
                    "The address of its `folo serve`; use https:// (a TLS proxy or a tunnel) across networks you \
                     do not control.",
                ),
                field(
                    "token",
                    "Token",
                    true,
                    false,
                    None,
                    "Its `--token` (or FOLO_TOKEN); empty when it runs without one.",
                ),
            ],
            provisions: false,
            self_terminates: false,
        },
    ]
}

/// RunPod's catalogue as the demo has it, cheapest first: both clouds, one GPU out of stock and one
/// low on it.
pub(super) fn runpod_offers() -> Vec<CloudOffer> {
    let gib = 1u64 << 30;
    let offer = |id: &str, gpu: &str, vram: u64, price: f64, availability: CloudAvailability| CloudOffer {
        id: id.into(),
        gpu: gpu.into(),
        gpu_count: 1,
        vram_bytes: vram * gib,
        price_per_hour: Some(price),
        region: None,
        detail: Some(if id.ends_with("@secure") { "Secure Cloud" } else { "Community Cloud" }.into()),
        availability,
    };
    vec![
        offer("NVIDIA RTX A4000@community", "RTX A4000", 16, 0.17, CloudAvailability::High),
        offer("NVIDIA GeForce RTX 3090@community", "RTX 3090", 24, 0.22, CloudAvailability::Medium),
        offer("NVIDIA GeForce RTX 4090@community", "RTX 4090", 24, 0.34, CloudAvailability::None),
        offer(RTX4090_OFFER, "RTX 4090", 24, 0.69, CloudAvailability::High),
        offer("NVIDIA RTX 6000 Ada Generation@secure", "RTX 6000 Ada", 48, 0.77, CloudAvailability::Low),
        offer(L40S_OFFER, "L40S", 48, 0.86, CloudAvailability::Medium),
        offer(A100_OFFER, "A100 80GB", 80, 1.64, CloudAvailability::Medium),
        offer("NVIDIA H100 80GB HBM3@secure", "H100 SXM", 80, 2.99, CloudAvailability::Low),
    ]
}

/// The demo's RunPod connection, checked 20 minutes ago.
pub(super) fn runpod_connection() -> CloudConnection {
    CloudConnection {
        id: RUNPOD_CONNECTION.into(),
        provider: "runpod".into(),
        name: "RunPod".into(),
        created: time::format(MOCK_NOW - 3 * 86_400),
        status: CloudConnectionStatus::Ok,
        account: Some("you@example.com".into()),
        balance_usd: Some(42.17),
        error: None,
        note: None,
        checked: Some(time::format(MOCK_NOW - 20 * 60)),
        values: [("api_key".to_string(), "••••a1b2".to_string())].into(),
    }
}

/// A secret as answers show it: `••••` and its last 4 characters.
pub(super) fn secret_hint(s: &str) -> String {
    let tail: String = s.chars().rev().take(4).collect::<Vec<_>>().into_iter().rev().collect();
    format!("••••{tail}")
}

/// An `http://` URL to a host that isn't loopback or a private address: the connection's `note`
/// warns about it.
pub(super) fn unencrypted(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("http://") else { return false };
    let host = rest.split(['/', ':']).next().unwrap_or_default();
    let private = ["localhost", "127.", "10.", "192.168.", "["].iter().any(|p| host.starts_with(p))
        || host
            .strip_prefix("172.")
            .and_then(|r| r.split('.').next()?.parse::<u8>().ok())
            .is_some_and(|b| (16..32).contains(&b));
    !private
}

/// What a connection can run on: RunPod's catalogue, or a remote server's own GPU (not billed).
/// A connection whose last check failed can't be asked.
pub(super) fn offers_of(conn: &CloudConnection) -> ApiResult<Vec<CloudOffer>> {
    if conn.status == CloudConnectionStatus::Error {
        return Err(ApiError::new("cloud_auth_failed", conn.error.clone().unwrap_or_default())
            .with_hint("Check the connection again in Settings › Cloud."));
    }
    if conn.provider == "runpod" {
        return Ok(runpod_offers());
    }
    Ok(vec![CloudOffer {
        id: "gpu:0".into(),
        gpu: "RTX 4080 SUPER".into(),
        gpu_count: 1,
        vram_bytes: 16 << 30,
        price_per_hour: None,
        region: None,
        detail: conn.account.clone(),
        availability: CloudAvailability::Unknown,
    }])
}

pub(super) fn no_connection(id: &str) -> ApiError {
    ApiError::new("connection_not_found", format!("No cloud connection {id}"))
        .with_hint("Settings › Cloud lists the connections.")
}

pub(super) fn no_instance(id: &str) -> ApiError {
    ApiError::new("instance_not_found", format!("No cloud instance {id}"))
        .with_hint("Settings › Cloud lists the instances.")
}

impl MockBackend {
    pub(super) fn connection(&self, id: &str) -> ApiResult<CloudConnection> {
        self.connections.lock().unwrap().iter().find(|c| c.id == id).cloned().ok_or_else(|| no_connection(id))
    }

    /// The connection and offer a run's `compute` names, checked as `POST /api/runs` checks them.
    pub(super) fn compute_offer(
        &self,
        c: &Compute,
        config: &TrainConfig,
    ) -> ApiResult<(CloudConnection, CloudOffer)> {
        if config.device != "auto" {
            return Err(ApiError::new(
                "invalid_config",
                format!(
                    "device must be auto for a cloud run (got {}): the worker picks its GPU",
                    config.device
                ),
            ));
        }
        if !(folo_api::MIN_MAX_HOURS..=folo_api::MAX_MAX_HOURS).contains(&c.max_hours) {
            return Err(ApiError::new(
                "invalid_config",
                format!("max_hours must be from 0.25 to 720 (got {})", c.max_hours),
            ));
        }
        let conn = self.connection(&c.connection_id)?;
        let offer = offers_of(&conn)?.into_iter().find(|o| o.id == c.offer_id).ok_or_else(|| {
            ApiError::new("offer_not_found", format!("{} has no offer {}", conn.name, c.offer_id))
                .with_hint("Offers change: pick one from the list again.")
        })?;
        Ok((conn, offer))
    }
}
