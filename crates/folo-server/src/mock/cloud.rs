//! The mock's cloud compute (docs/API.md, Mock mode → Cloud): the real providers, one RunPod connection
//! ([`CONNECTION`]) with a fixed list of offers, and cloud runs whose hand-over (provisioning, booting and syncing
//! before training, collecting and releasing after it) and instance are pure functions of the run's timeline and the
//! clock, as the mock's runs are, so the run, its stream and `/api/cloud/instances` agree at any instant.

use std::collections::BTreeMap;

use axum::http::StatusCode;
use folo_api::{
    CloudAvailability, CloudConnection, CloudConnectionStatus, CloudInstance, CloudInstanceStatus,
    CloudOffer, CloudPhase, CloudProvider, ConnectCloud, RunCloud, RunStatus, TransferProgress,
};

use super::runs::Snap;
use crate::error::{Error, Result};
use crate::util::rfc3339;

/// Environment variable that picks how much of the cloud the mock serves ([`MockCloud`]).
pub const ENV_MOCK_CLOUD: &str = "FOLO_MOCK_CLOUD";
/// The demo's RunPod connection.
pub const CONNECTION: &str = "cc_4d2a";
/// The offer the seeded cloud run (`run_3c01`) rented.
pub const SEEDED_OFFER: &str = "NVIDIA GeForce RTX 4090@secure";
/// Seconds of each phase around the training: provisioning, booting and syncing before it, collecting and releasing
/// after it.
pub const PHASE_S: f64 = 2.0;
/// What the simulated upload sends and the collection copies back.
const UPLOAD_BYTES: u64 = 48_000_000;
const WEIGHTS_BYTES: u64 = 21_000_000;

/// How much of the cloud the mock serves ([`ENV_MOCK_CLOUD`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MockCloud {
    /// Everything, with the demo connection.
    #[default]
    On,
    /// The `cloud` feature without a connection (`none`): the empty states of the front ends.
    NoConnection,
    /// No `cloud` feature (`off`): a server from before cloud compute.
    Off,
}

impl MockCloud {
    /// [`ENV_MOCK_CLOUD`], or [`MockCloud::On`] when it is unset or unrecognised.
    pub fn from_env() -> MockCloud {
        Self::parse(&std::env::var(ENV_MOCK_CLOUD).unwrap_or_default())
    }

    /// An [`ENV_MOCK_CLOUD`] value (case-insensitive): `none`, `off`, anything else is [`MockCloud::On`].
    pub fn parse(s: &str) -> MockCloud {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => MockCloud::NoConnection,
            "off" => MockCloud::Off,
            _ => MockCloud::On,
        }
    }
}

/// `501 not_implemented` of `/api/cloud` with [`MockCloud::Off`].
pub fn off() -> Error {
    Error::not_implemented("cloud compute").hint(format!("the mock serves it unless {ENV_MOCK_CLOUD}=off"))
}

fn offer(
    id: &str,
    gpu: &str,
    vram_gib: u64,
    price: f64,
    detail: &str,
    availability: CloudAvailability,
) -> CloudOffer {
    CloudOffer {
        id: id.into(),
        gpu: gpu.into(),
        gpu_count: 1,
        vram_bytes: vram_gib << 30,
        price_per_hour: Some(price),
        region: None,
        detail: Some(detail.into()),
        availability,
    }
}

/// RunPod's offers, cheapest first: both clouds, one with nothing available now.
pub fn offers() -> Vec<CloudOffer> {
    use CloudAvailability::{High, Low, Medium, None};
    vec![
        offer("NVIDIA RTX A4000@community", "RTX A4000", 16, 0.17, "Community Cloud", High),
        offer("NVIDIA GeForce RTX 3090@community", "RTX 3090", 24, 0.22, "Community Cloud", Medium),
        offer("NVIDIA GeForce RTX 4090@community", "RTX 4090", 24, 0.34, "Community Cloud", Low),
        offer(SEEDED_OFFER, "RTX 4090", 24, 0.69, "Secure Cloud", High),
        offer("NVIDIA L40S@secure", "L40S", 48, 0.86, "Secure Cloud", Medium),
        offer("NVIDIA A100 80GB PCIe@secure", "A100 PCIe", 80, 1.64, "Secure Cloud", None),
        offer("NVIDIA H100 80GB HBM3@secure", "H100 SXM", 80, 2.99, "Secure Cloud", High),
    ]
}

/// The demo connection, checked a few minutes before `now`.
pub fn seeded(now: f64) -> CloudConnection {
    CloudConnection {
        id: CONNECTION.into(),
        provider: "runpod".into(),
        name: "RunPod".into(),
        created: rfc3339(now - 12.0 * 86_400.0),
        status: CloudConnectionStatus::Ok,
        account: Some("you@example.com".into()),
        balance_usd: Some(42.17),
        error: None,
        note: None,
        checked: Some(rfc3339(now - 180.0)),
        values: BTreeMap::from([("api_key".to_string(), "••••a1b2".to_string())]),
    }
}

/// A new connection of `req` to `provider`: any values do, except a secret containing `bad`, which the "provider"
/// rejects. A required field must be there, as for the real store.
pub fn connect(
    req: ConnectCloud,
    provider: &CloudProvider,
    id: String,
    taken: &[CloudConnection],
    now: f64,
) -> Result<CloudConnection> {
    let mut values = BTreeMap::new();
    for f in &provider.fields {
        let v = req.values.get(&f.key).map(|v| v.trim()).unwrap_or_default();
        if v.is_empty() {
            if f.required {
                return Err(Error::new(
                    StatusCode::BAD_REQUEST,
                    "invalid_body",
                    format!("`{}` is required", f.key),
                ));
            }
            continue;
        }
        if f.secret && v.contains("bad") {
            return Err(Error::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "cloud_auth_failed",
                format!("{} rejected the {}", provider.name, f.label.to_lowercase()),
            )
            .hint("check the credentials in the provider's console, then connect again"));
        }
        let shown = if f.secret { crate::cloud::redact(v) } else { v.to_string() };
        values.insert(f.key.clone(), shown);
    }
    let name = match req.name.trim() {
        "" => (1..)
            .map(|i| if i == 1 { provider.name.clone() } else { format!("{} {i}", provider.name) })
            .find(|n| !taken.iter().any(|c| &c.name == n))
            .expect("some suffix is free"),
        n => n.to_string(),
    };
    let remote = provider.id == "remote";
    Ok(CloudConnection {
        id,
        provider: provider.id.clone(),
        name,
        created: rfc3339(now),
        status: CloudConnectionStatus::Ok,
        account: Some(if remote { "gpu-box.example.com".into() } else { "you@example.com".into() }),
        balance_usd: (!remote).then_some(42.17),
        error: None,
        note: None,
        checked: Some(rfc3339(now)),
        values,
    })
}

/// A simulated cloud run's hand-over: where it runs, and the instance it has.
#[derive(Clone, Debug)]
pub struct SimCloud {
    /// Its connection.
    pub connection_id: String,
    /// The offer it rented.
    pub offer: CloudOffer,
    /// `compute.max_hours`.
    pub max_hours: f32,
    /// Its instance's id (`ci_<run suffix>`) and name at the provider.
    pub instance_id: String,
    name: String,
    provider_ref: String,
}

impl SimCloud {
    /// The hand-over of run `run_id` on `offer`.
    pub fn new(run_id: &str, connection_id: &str, offer: CloudOffer, max_hours: f32) -> SimCloud {
        let suffix = run_id.strip_prefix("run_").unwrap_or(run_id);
        let h = crate::util::fnv1a(run_id.as_bytes());
        SimCloud {
            connection_id: connection_id.into(),
            offer,
            max_hours,
            instance_id: format!("ci_{suffix}"),
            name: format!("folo-{run_id}-{:04x}", h & 0xffff),
            provider_ref: format!("{:014x}", h >> 8),
        }
    }

    /// `Run.cloud` at `snap`, for a run whose training ends `trained` (Unix seconds, once it started) and whose
    /// epochs take `epoch_s` (`done` validated so far).
    pub fn at(&self, snap: &Snap, trained: Option<f64>, epoch_s: f64, done: u32) -> RunCloud {
        let launched = snap.created;
        let t = snap.at - launched;
        let part = |from: f64, bytes: u64| {
            let tenths = ((snap.at - from) / PHASE_S * 10.0).floor().clamp(0.0, 10.0) as u64;
            Some(TransferProgress { done_bytes: bytes * tenths / 10, total_bytes: bytes })
        };
        let (phase, progress) = match snap.status {
            RunStatus::Queued if t < PHASE_S => (CloudPhase::Provisioning, None),
            RunStatus::Queued if t < 2.0 * PHASE_S => (CloudPhase::Booting, None),
            RunStatus::Queued => (CloudPhase::Syncing, part(launched + 2.0 * PHASE_S, UPLOAD_BYTES)),
            RunStatus::Stopping => (CloudPhase::Releasing, None),
            RunStatus::Running => match trained {
                Some(end) if snap.at >= end + PHASE_S => (CloudPhase::Releasing, None),
                Some(end) if snap.at >= end => (CloudPhase::Collecting, part(end, WEIGHTS_BYTES)),
                _ => (CloudPhase::Training, None),
            },
            _ => (CloudPhase::Released, None),
        };
        let end = snap.finished.unwrap_or(snap.at);
        let cost = self.offer.price_per_hour.map(|p| ((end - launched) / 3600.0 * p * 100.0).round() / 100.0);
        let synced = (done > 0).then(|| snap.started.map(|s| rfc3339(s + done as f64 * epoch_s))).flatten();
        // Where the hand-over was when the run ended: the collection after training, training cut short, or the upload
        // of a run stopped before it trained.
        let ended_in = matches!(phase, CloudPhase::Releasing | CloudPhase::Released).then(|| {
            match (snap.started, trained) {
                (_, Some(end)) if snap.at >= end => CloudPhase::Collecting,
                (Some(_), _) => CloudPhase::Training,
                (None, _) => CloudPhase::Syncing,
            }
        });
        RunCloud {
            connection_id: self.connection_id.clone(),
            provider: "runpod".into(),
            offer: self.offer.clone(),
            instance_id: Some(self.instance_id.clone()),
            phase,
            progress,
            max_hours: self.max_hours,
            deadline: Some(rfc3339(launched + self.max_hours as f64 * 3600.0)),
            cost_usd: cost,
            synced,
            ended_in,
            unanswered_since: None,
        }
    }

    /// Its instance as `cloud` (the run's [`SimCloud::at`]) says.
    pub fn instance(&self, run_id: &str, snap: &Snap, cloud: &RunCloud) -> CloudInstance {
        let status = match cloud.phase {
            CloudPhase::Provisioning => CloudInstanceStatus::Launching,
            CloudPhase::Booting => CloudInstanceStatus::Starting,
            CloudPhase::Releasing => CloudInstanceStatus::Terminating,
            CloudPhase::Released | CloudPhase::Unknown => CloudInstanceStatus::Terminated,
            _ => CloudInstanceStatus::Running,
        };
        CloudInstance {
            id: self.instance_id.clone(),
            connection_id: self.connection_id.clone(),
            provider: "runpod".into(),
            provider_ref: (status != CloudInstanceStatus::Launching).then(|| self.provider_ref.clone()),
            name: self.name.clone(),
            run_id: Some(run_id.into()),
            offer: self.offer.clone(),
            status,
            created: rfc3339(snap.created),
            ended: (!status.is_alive()).then(|| snap.finished.map(rfc3339)).flatten(),
            deadline: cloud.deadline.clone(),
            cost_usd: cloud.cost_usd,
            error: None,
        }
    }

    /// The log line of the start of `phase`, as the real control plane writes it (`cost` of the released run).
    pub fn phase_line(&self, phase: CloudPhase, cost: Option<f64>) -> Option<String> {
        Some(match phase {
            CloudPhase::Provisioning => format!("launching {} at RunPod", self.offer.gpu),
            CloudPhase::Booting => "waiting for the instance to start".into(),
            CloudPhase::Syncing => "uploading the dataset".into(),
            CloudPhase::Training => "the worker took the run".into(),
            CloudPhase::Collecting => "the run ended on the worker; copying its weights".into(),
            CloudPhase::Releasing => "terminating the instance".into(),
            CloudPhase::Released => match cost {
                Some(c) => format!("terminated the instance; it cost about ${c:.2}"),
                None => "terminated the instance".into(),
            },
            CloudPhase::Unknown => return None,
        })
    }
}
