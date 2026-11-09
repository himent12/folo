//! Cloud compute (docs/CLOUD.md; docs/API.md, *Cloud compute* and *Uploads*): providers, connections, offers and
//! instances of `/api/cloud`, a run's `compute` request and `cloud` state, and the upload sessions a control plane
//! sends datasets and weights to a worker with.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Dataset, DatasetFormat, Model};

/// [`SystemResponse::features`](crate::SystemResponse::features): `/api/cloud` exists, `TrainConfig.compute` is
/// honoured, runs carry `cloud` and streams send `cloud` events.
pub const FEATURE_CLOUD: &str = "cloud";
/// [`SystemResponse::features`](crate::SystemResponse::features): `/api/uploads` exists.
pub const FEATURE_UPLOADS: &str = "uploads";
/// [`SystemResponse::features`](crate::SystemResponse::features): `/metrics` takes `after_step` / `after_epoch`,
/// `/logs` takes `after` and answers `total`, and `/weights` answers `Range` requests.
pub const FEATURE_RUN_CURSORS: &str = "run_cursors";
/// [`SystemResponse::features`](crate::SystemResponse::features): `POST /api/runs` takes `checkpoint`.
pub const FEATURE_RUN_CHECKPOINT: &str = "run_checkpoint";

/// Default of [`Compute::max_hours`].
pub const DEFAULT_MAX_HOURS: f32 = 12.0;
/// Smallest [`Compute::max_hours`] (15 minutes).
pub const MIN_MAX_HOURS: f32 = 0.25;
/// Largest [`Compute::max_hours`] (30 days).
pub const MAX_MAX_HOURS: f32 = 720.0;

// ---------------------------------------------------------------------------------------------------------------
// Providers and connections
// ---------------------------------------------------------------------------------------------------------------

/// `GET /api/cloud/providers`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudProviderList {
    /// Every provider this server knows, in display order.
    pub providers: Vec<CloudProvider>,
}

/// A kind of compute FOLO can use, with the fields a connection to it takes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudProvider {
    /// Stable id: `runpod`, `remote`.
    pub id: String,
    /// Display name, e.g. `RunPod`.
    pub name: String,
    /// One sentence on what it is.
    pub about: String,
    /// Where its credentials are made (a page of the provider's console or docs), if there is one.
    #[serde(default)]
    pub docs_url: Option<String>,
    /// What a connection asks for, in form order.
    pub fields: Vec<CloudField>,
    /// It launches an instance per run and terminates it afterwards (`false`: it uses a machine that already runs).
    pub provisions: bool,
    /// Its instances terminate themselves after a run's deadline, also while this server is off.
    pub self_terminates: bool,
}

/// One field of a connection ([`CloudProvider::fields`], [`ConnectCloud::values`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudField {
    /// Key in [`ConnectCloud::values`], e.g. `api_key`.
    pub key: String,
    /// Form label, e.g. `API key`.
    pub label: String,
    /// A credential: entered masked, never returned (connections show only a hint of it).
    pub secret: bool,
    /// A connection cannot be made without it.
    pub required: bool,
    /// Example value.
    #[serde(default)]
    pub placeholder: Option<String>,
    /// One line of help.
    #[serde(default)]
    pub help: Option<String>,
}

/// `GET /api/cloud/connections`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudConnectionList {
    /// Connections, oldest first.
    pub connections: Vec<CloudConnection>,
}

/// `POST /api/cloud/connections` body.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConnectCloud {
    /// Provider id ([`CloudProvider::id`]).
    pub provider: String,
    /// Display name; empty means the provider's name (with ` 2`, ` 3`, ... when taken).
    #[serde(default)]
    pub name: String,
    /// A value per [`CloudField::key`]; empty values count as missing.
    #[serde(default)]
    pub values: BTreeMap<String, String>,
}

// Not derived: `values` holds credentials, which must not reach a log through `{:?}`.
impl std::fmt::Debug for ConnectCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectCloud")
            .field("provider", &self.provider)
            .field("name", &self.name)
            .field("values", &self.values.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Whether a connection's credentials work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum CloudConnectionStatus {
    /// The last check passed.
    #[default]
    Ok,
    /// The last check failed; `error` says why.
    Error,
}

/// A provider with credentials.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudConnection {
    /// Opaque id, e.g. `cc_1f2e`.
    pub id: String,
    /// Provider id.
    pub provider: String,
    /// Display name.
    pub name: String,
    /// RFC 3339 creation time.
    pub created: String,
    /// Result of the last check.
    pub status: CloudConnectionStatus,
    /// Who the credentials belong to, as the provider names them (an email, a host); `null` when unknown.
    #[serde(default)]
    pub account: Option<String>,
    /// The account's balance in US dollars, where the provider reports one.
    #[serde(default)]
    pub balance_usd: Option<f64>,
    /// Why the last check failed (`status: error`).
    #[serde(default)]
    pub error: Option<String>,
    /// Something the user should know about a working connection (e.g. an unencrypted `http://` URL).
    #[serde(default)]
    pub note: Option<String>,
    /// RFC 3339 time of the last check.
    #[serde(default)]
    pub checked: Option<String>,
    /// The connection's values by [`CloudField::key`]: fields that are not secret as entered, secrets as a hint
    /// (`••••` and their last 4 characters).
    #[serde(default)]
    pub values: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------------------------------------------
// Offers and instances
// ---------------------------------------------------------------------------------------------------------------

/// `GET /api/cloud/connections/{id}/offers`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudOfferList {
    /// What the connection can run on, cheapest first (offers without a price first).
    pub offers: Vec<CloudOffer>,
    /// RFC 3339 time the provider was asked (answers are cached for a minute).
    pub fetched: String,
}

/// How likely an offer is to start now.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum CloudAvailability {
    /// Plenty.
    High,
    /// Some.
    Medium,
    /// Few left.
    Low,
    /// None right now: launching would be refused.
    None,
    /// The provider does not say.
    #[default]
    Unknown,
}

/// Something a connection can run on: a GPU type with its price.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudOffer {
    /// Opaque id, unique within the connection (`Compute.offer_id`).
    pub id: String,
    /// GPU name, e.g. `RTX 4090`.
    pub gpu: String,
    /// GPUs of the instance (a run trains on one).
    #[serde(default = "one")]
    pub gpu_count: u32,
    /// Memory of one GPU in bytes (0 when unknown).
    #[serde(default)]
    pub vram_bytes: u64,
    /// US dollars per hour; `null` for compute that is not billed (your own machine).
    #[serde(default)]
    pub price_per_hour: Option<f64>,
    /// Region or data centre, when the provider names one.
    #[serde(default)]
    pub region: Option<String>,
    /// A qualifier of the offer, e.g. `Secure Cloud` or `Community Cloud`.
    #[serde(default)]
    pub detail: Option<String>,
    /// How likely it is to start now.
    #[serde(default)]
    pub availability: CloudAvailability,
}

fn one() -> u32 {
    1
}

/// `GET /api/cloud/instances`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudInstanceList {
    /// Instances that are alive, then those that ended in the last 7 days, newest first.
    pub instances: Vec<CloudInstance>,
}

/// Where an instance is in its life.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum CloudInstanceStatus {
    /// The provider is being asked for it.
    Launching,
    /// The provider has it; its worker does not answer yet.
    Starting,
    /// Its worker answers.
    Running,
    /// Termination was requested and is not confirmed yet (`error` says why when requests fail; they are retried).
    Terminating,
    /// The provider confirmed it is gone.
    Terminated,
    /// The provider no longer has it although FOLO did not terminate it (it failed, or was removed elsewhere).
    Lost,
}

impl CloudInstanceStatus {
    /// True until the instance is `terminated` or `lost`: it may still cost money.
    pub fn is_alive(self) -> bool {
        !matches!(self, CloudInstanceStatus::Terminated | CloudInstanceStatus::Lost)
    }
}

/// A machine FOLO launched at a provider.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CloudInstance {
    /// Opaque id, e.g. `ci_8a2f`.
    pub id: String,
    /// The connection it was launched with.
    pub connection_id: String,
    /// Provider id.
    pub provider: String,
    /// The provider's own id of it (a RunPod pod id), to find it in the provider's console; `null` while
    /// `launching`.
    #[serde(default)]
    pub provider_ref: Option<String>,
    /// Its name at the provider, `folo-<run id>-<nonce>`.
    pub name: String,
    /// The run it was launched for.
    #[serde(default)]
    pub run_id: Option<String>,
    /// What it is.
    pub offer: CloudOffer,
    /// Where it is in its life.
    pub status: CloudInstanceStatus,
    /// RFC 3339 time of the launch request.
    pub created: String,
    /// RFC 3339 time it was confirmed gone.
    #[serde(default)]
    pub ended: Option<String>,
    /// RFC 3339 time it is terminated at the latest (the run's `max_hours`).
    #[serde(default)]
    pub deadline: Option<String>,
    /// Estimated cost so far in US dollars (`price_per_hour` times its lifetime); `null` without a price.
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// The last failure of a provider request about it (e.g. a termination that is being retried).
    #[serde(default)]
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------------------------------------------

/// `TrainConfig.compute`: train on a cloud connection instead of this machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Compute {
    /// The connection to run on.
    pub connection_id: String,
    /// One of its offers.
    pub offer_id: String,
    /// The run is stopped and its instance terminated after this many hours at the latest (0.25 to 720).
    #[serde(default = "default_max_hours")]
    pub max_hours: f32,
    /// Disk of the instance in GB; `null` sizes it for the dataset (twice its size plus 20 GB).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disk_gb: Option<u32>,
}

fn default_max_hours() -> f32 {
    DEFAULT_MAX_HOURS
}

/// Where a cloud run is in its lifecycle (docs/CLOUD.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum CloudPhase {
    /// Asking the provider for an instance.
    Provisioning,
    /// Waiting for the instance and its worker.
    Booting,
    /// Uploading the dataset and weights.
    Syncing,
    /// The worker trains.
    Training,
    /// Copying the final weights back.
    Collecting,
    /// Terminating the instance.
    Releasing,
    /// Done: nothing of the run is left at the provider.
    Released,
}

impl CloudPhase {
    /// Every phase in order.
    pub const ALL: [CloudPhase; 7] = [
        CloudPhase::Provisioning,
        CloudPhase::Booting,
        CloudPhase::Syncing,
        CloudPhase::Training,
        CloudPhase::Collecting,
        CloudPhase::Releasing,
        CloudPhase::Released,
    ];

    /// The wire spelling, e.g. `"provisioning"`.
    pub fn as_str(self) -> &'static str {
        match self {
            CloudPhase::Provisioning => "provisioning",
            CloudPhase::Booting => "booting",
            CloudPhase::Syncing => "syncing",
            CloudPhase::Training => "training",
            CloudPhase::Collecting => "collecting",
            CloudPhase::Releasing => "releasing",
            CloudPhase::Released => "released",
        }
    }
}

/// Bytes moved of a transfer ([`RunCloud::progress`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TransferProgress {
    /// Bytes sent or received so far.
    pub done_bytes: u64,
    /// Bytes in total (0 when unknown).
    pub total_bytes: u64,
}

/// `Run.cloud` and the SSE `cloud` event: where a cloud run executes and how far it is.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RunCloud {
    /// The connection it runs on.
    pub connection_id: String,
    /// Provider id.
    pub provider: String,
    /// The offer it runs on, as it was when the run started.
    pub offer: CloudOffer,
    /// Its instance (`GET /api/cloud/instances`); `null` before one is launched and for providers that launch none.
    #[serde(default)]
    pub instance_id: Option<String>,
    /// Where it is.
    pub phase: CloudPhase,
    /// Progress of the phase's transfer (`syncing`, `collecting`), else `null`.
    #[serde(default)]
    pub progress: Option<TransferProgress>,
    /// The run's time limit ([`Compute::max_hours`]).
    pub max_hours: f32,
    /// RFC 3339 time the limit ends; `null` until the instance exists.
    #[serde(default)]
    pub deadline: Option<String>,
    /// Estimated cost so far in US dollars (final once `released`); `null` without a price.
    #[serde(default)]
    pub cost_usd: Option<f64>,
    /// RFC 3339 time `weights/last.folo` was last copied to this machine.
    #[serde(default)]
    pub synced: Option<String>,
}

/// `RunEstimate.cost`: what a cloud run would cost.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CostEstimate {
    /// The offer's price, US dollars per hour.
    pub price_per_hour: f64,
    /// `price_per_hour` times `est_total_s` (training only: booting and the upload come on top).
    pub est_usd: f64,
    /// The same for `est_total_s_low`.
    pub est_usd_low: f64,
    /// The same for `est_total_s_high`.
    pub est_usd_high: f64,
    /// The most the run can cost: `price_per_hour` times `compute.max_hours`.
    pub max_usd: f64,
}

// ---------------------------------------------------------------------------------------------------------------
// Uploads
// ---------------------------------------------------------------------------------------------------------------

/// What an upload becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum UploadKind {
    /// A tar archive of a dataset folder; registered as a dataset.
    Dataset,
    /// A `.folo` file; listed as a model.
    Model,
}

/// `POST /api/uploads` body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CreateUpload {
    /// What is uploaded.
    pub kind: UploadKind,
    /// Name of the dataset or model.
    pub name: String,
    /// Dataset format (`kind: dataset`).
    #[serde(default)]
    pub format: DatasetFormat,
    /// An opaque identity of the content (at most 128 characters). When the server already holds an upload of
    /// this kind with it, it answers with that (`existing`) and nothing needs to be sent.
    #[serde(default)]
    pub fingerprint: Option<String>,
    /// Total bytes of the stream, when known (for progress; not checked).
    #[serde(default)]
    pub bytes: Option<u64>,
}

/// An upload session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Upload {
    /// Opaque id, e.g. `up_5c1d`.
    pub id: String,
    /// What is uploaded.
    pub kind: UploadKind,
    /// Name of the dataset or model.
    pub name: String,
    /// Index of the next part the server expects (0-based).
    pub part: u32,
    /// Bytes received so far.
    pub received_bytes: u64,
    /// Largest part the server takes, in bytes.
    pub max_part_bytes: u64,
    /// RFC 3339 time the session is dropped unless a part arrives first.
    pub expires: String,
}

/// `POST /api/uploads` response: a session to send parts to, or what the server already has for the fingerprint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct UploadCreated {
    /// The new session (`201`); `null` when `existing` is set.
    #[serde(default)]
    pub upload: Option<Upload>,
    /// The dataset or model already uploaded with this fingerprint (`200`).
    #[serde(default)]
    pub existing: Option<Uploaded>,
}

/// What an upload became (`POST /api/uploads/{id}/complete`): exactly one of the two.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Uploaded {
    /// The registered dataset (`kind: dataset`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<Dataset>,
    /// The listed model (`kind: model`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<Model>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connect_debug_hides_values() {
        let c = ConnectCloud {
            provider: "runpod".into(),
            name: String::new(),
            values: BTreeMap::from([("api_key".to_string(), "rpa_SECRET".to_string())]),
        };
        let shown = format!("{c:?}");
        assert!(shown.contains("api_key") && !shown.contains("SECRET"), "{shown}");
    }

    #[test]
    fn compute_defaults_and_phase_spellings() {
        let c: Compute = serde_json::from_str(r#"{"connection_id":"cc_1","offer_id":"o"}"#).unwrap();
        assert_eq!((c.max_hours, c.disk_gb), (DEFAULT_MAX_HOURS, None));
        assert!(!serde_json::to_string(&c).unwrap().contains("disk_gb"));
        for p in CloudPhase::ALL {
            assert_eq!(serde_json::to_value(p).unwrap(), p.as_str());
        }
        assert!(CloudPhase::Syncing < CloudPhase::Training);
        assert!(CloudInstanceStatus::Terminating.is_alive() && !CloudInstanceStatus::Lost.is_alive());
    }
}
