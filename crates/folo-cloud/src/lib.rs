//! Cloud compute providers (docs/CLOUD.md).
//!
//! A [`Provider`] is a kind of compute FOLO can train on: it checks a connection's credentials, lists what the
//! connection can run on ([`folo_api::CloudOffer`]) and, when it provisions, launches, looks up and terminates
//! instances whose only job is to run `folo serve` (the *worker*, started by [`bootstrap`]). Everything after
//! that (uploading the dataset, starting the run, mirroring it, collecting the weights, the instance ledger) is
//! provider-agnostic and lives in `folo_server::cloud`, which talks to the worker through `folo-client`.
//!
//! This crate has no engine and no server dependencies: a provider is HTTP calls and a little mapping.
//!
//! Providers: [`runpod::RunPod`] (GPU pods rented per run) and [`remote::Remote`] (a FOLO server that already
//! runs). [`providers`] lists them; adding one is a type implementing [`Provider`] and a line there.

use std::collections::BTreeMap;
use std::sync::Arc;

use folo_api::{CloudOffer, CloudProvider};

pub mod bootstrap;
#[cfg(test)]
mod fixture;
mod http;
pub mod remote;
pub mod runpod;

/// The port a worker's `folo serve` listens on inside its instance.
pub const WORKER_PORT: u16 = 7878;

/// Overrides where workers download `install.sh` and the release archive from ([`worker_base_url`]).
pub const WORKER_BASE_URL_ENV: &str = "FOLO_CLOUD_WORKER_BASE_URL";

/// The connection field with the same job as [`WORKER_BASE_URL_ENV`], for one connection.
pub const WORKER_BASE_URL_FIELD: &str = "worker_base_url";

/// Where a worker of this FOLO version downloads `install.sh` and `folo-<target>.tar.gz` from: the connection's
/// `worker_base_url`, else `$FOLO_CLOUD_WORKER_BASE_URL`, else the GitHub release of `version` (the version of the
/// control plane, so both ends speak the same API). Without a trailing slash.
pub fn worker_base_url(values: &Values, version: &str) -> String {
    let from_env = std::env::var(WORKER_BASE_URL_ENV).ok();
    let chosen = values.get(WORKER_BASE_URL_FIELD).map(str::to_string).or(from_env);
    match chosen.as_deref().map(|u| u.trim().trim_end_matches('/')) {
        Some(u) if !u.is_empty() => u.to_string(),
        _ => format!("https://github.com/himent12/folo/releases/download/v{version}"),
    }
}

/// The platform every worker runs on: the release archive an instance installs is `folo-<this>.tar.gz`.
pub const WORKER_TARGET: &str = "x86_64-unknown-linux-gnu";

/// Whether an instance could install FOLO from a place ([`worker_download`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerDownload {
    /// `install.sh` and the archive's checksum are served there.
    Found,
    /// The place answers and says a file is not there (a release that is not published, a private repository, a
    /// mistyped URL: any `4xx` but those that blame the moment, `408`, `425` and `429`): an instance launched now
    /// would fail its setup.
    Missing {
        /// The file asked for.
        url: String,
        /// The HTTP status it was answered with.
        status: u16,
    },
    /// This machine got no usable answer (no connection, a server error). The instance may still reach the place
    /// (a mirror inside its network), so this is not a reason to refuse a launch.
    Unknown(String),
}

impl WorkerDownload {
    /// What to tell the user when the download is [`WorkerDownload::Missing`]: what answered what, and the way
    /// out.
    pub fn problem(&self) -> Option<String> {
        let WorkerDownload::Missing { url, status } = self else { return None };
        Some(format!(
            "instances cannot install FOLO: {url} answers {status} (the release is not published, or its \
             repository is private). Give the connection a worker download URL, or set {WORKER_BASE_URL_ENV}"
        ))
    }
}

/// Asks `base_url` ([`worker_base_url`]) for the two small files every install starts with, `install.sh` and the
/// archive's `.sha256`, as an instance would: without credentials. Two short requests.
pub fn worker_download(base_url: &str) -> WorkerDownload {
    let base = base_url.trim_end_matches('/');
    for file in ["install.sh".to_string(), format!("folo-{WORKER_TARGET}.tar.gz.sha256")] {
        let url = format!("{base}/{file}");
        match http::status_of(&url) {
            Ok(200..=299) => {}
            // A refusal of the moment (a timeout, a rate limit on this machine's address) says nothing about what
            // the instance will get.
            Ok(status @ (408 | 425 | 429)) => {
                return WorkerDownload::Unknown(format!("{url} answers {status}"));
            }
            Ok(status @ 400..=499) => return WorkerDownload::Missing { url, status },
            Ok(status) => return WorkerDownload::Unknown(format!("{url} answers {status}")),
            Err(e) => return WorkerDownload::Unknown(format!("{url} does not answer ({e})")),
        }
    }
    WorkerDownload::Found
}

/// What went wrong with a provider request, by who has to act.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// The provider rejected the credentials (`cloud_auth_failed`).
    Auth,
    /// The provider understood and said no: no capacity for the offer, no balance, a quota (`cloud_refused`).
    Refused,
    /// The provider's API does not answer, or answers with a server error or something unexpected
    /// (`cloud_unavailable`). Worth trying again.
    Unavailable,
    /// The connection has no such offer (`offer_not_found`).
    NoOffer,
    /// A connection value is missing or malformed (`invalid_body`).
    Invalid,
    /// An instance could not become a worker: FOLO cannot be installed on it (`cloud_boot_failed`). Nothing was
    /// launched.
    NoWorker,
}

/// A failed provider request. The message never contains a credential.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Who has to act.
    pub kind: ErrorKind,
    /// What happened, for people.
    pub message: String,
}

impl Error {
    /// An error of `kind`.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Error {
        Error { kind, message: message.into() }
    }

    /// The stable API code (`folo_api::codes`) this error is reported under.
    pub fn code(&self) -> &'static str {
        match self.kind {
            ErrorKind::Auth => "cloud_auth_failed",
            ErrorKind::Refused => "cloud_refused",
            ErrorKind::Unavailable => "cloud_unavailable",
            ErrorKind::NoOffer => "offer_not_found",
            ErrorKind::Invalid => "invalid_body",
            ErrorKind::NoWorker => "cloud_boot_failed",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for Error {}

/// `Result` of a provider call.
pub type Result<T> = std::result::Result<T, Error>;

/// A connection's values by [`folo_api::CloudField::key`]: credentials and settings. Empty values count as
/// missing.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Values(BTreeMap<String, String>);

// Not derived: values are credentials, which must not reach a log through `{:?}`.
impl std::fmt::Debug for Values {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.0.keys()).finish()
    }
}

impl From<BTreeMap<String, String>> for Values {
    fn from(map: BTreeMap<String, String>) -> Values {
        Values(
            map.into_iter().map(|(k, v)| (k, v.trim().to_string())).filter(|(_, v)| !v.is_empty()).collect(),
        )
    }
}

impl Values {
    /// The value of `key`, if it was given.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    /// The value of `key`, or an [`ErrorKind::Invalid`] error naming it.
    pub fn require(&self, key: &str) -> Result<&str> {
        self.get(key).ok_or_else(|| Error::new(ErrorKind::Invalid, format!("`{key}` is required")))
    }

    /// Every value, for the connection file.
    pub fn as_map(&self) -> &BTreeMap<String, String> {
        &self.0
    }
}

/// Who a connection's credentials belong to ([`Provider::check`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Account {
    /// How the provider names the account (an email, a host).
    pub label: Option<String>,
    /// Its balance in US dollars, where the provider reports one.
    pub balance_usd: Option<f64>,
    /// Something the user should know about a working connection (an unencrypted URL).
    pub note: Option<String>,
}

/// A FOLO server and the token it takes.
#[derive(Clone, PartialEq, Eq)]
pub struct Endpoint {
    /// Base URL, e.g. `https://abc123-7878.proxy.runpod.net`.
    pub url: String,
    /// Its `--token` (empty: none).
    pub token: String,
}

// Not derived: the token must not reach a log through `{:?}`.
impl std::fmt::Debug for Endpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Endpoint").field("url", &self.url).finish_non_exhaustive()
    }
}

impl Endpoint {
    /// A client of the worker.
    pub fn client(&self) -> folo_client::Client {
        folo_client::Client::new(&self.url).token(self.token.clone())
    }
}

/// What to launch ([`Provider::launch`]).
#[derive(Clone, PartialEq, Eq)]
pub struct LaunchSpec {
    /// The instance's name at the provider, unique to this launch (`folo-<run id>-<nonce>`): [`Provider::find`]
    /// looks instances up by it after a crash.
    pub name: String,
    /// The offer to launch ([`CloudOffer::id`]).
    pub offer_id: String,
    /// Disk in GB.
    pub disk_gb: u32,
    /// The token the worker's `folo serve` must require.
    pub token: String,
    /// The instance terminates itself this many seconds after it starts, where the provider can
    /// ([`CloudProvider::self_terminates`]).
    pub lifetime_s: u64,
    /// Where the worker downloads FOLO from ([`worker_base_url`]).
    pub worker_base_url: String,
}

// Not derived: the token must not reach a log through `{:?}`.
impl std::fmt::Debug for LaunchSpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LaunchSpec")
            .field("name", &self.name)
            .field("offer_id", &self.offer_id)
            .field("disk_gb", &self.disk_gb)
            .field("lifetime_s", &self.lifetime_s)
            .field("worker_base_url", &self.worker_base_url)
            .finish_non_exhaustive()
    }
}

/// Where an instance is, as the provider sees it ([`Provider::state`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstanceState {
    /// The provider has it and it is not reachable yet.
    Starting,
    /// It runs and its worker port is at `url` (the worker itself may still be installing: the caller polls
    /// `/api/health`).
    Running {
        /// Base URL of the worker's API.
        url: String,
    },
    /// The provider still has it, but it no longer runs and will not come back by itself (its container exited or
    /// failed). It may still be billed for its disk: terminate it.
    Stopped,
    /// The provider no longer has it (terminated, or it never existed).
    Gone,
}

/// A kind of compute. Calls block (HTTP requests with timeouts) and are safe from any thread.
///
/// Errors never contain a credential, and implementations never log one.
pub trait Provider: Send + Sync {
    /// Its id, name and connection fields.
    fn info(&self) -> CloudProvider;

    /// Checks `values` against the provider: [`ErrorKind::Invalid`] when one is missing or malformed (before
    /// any request), [`ErrorKind::Auth`] when the provider rejects them.
    fn check(&self, values: &Values) -> Result<Account>;

    /// What the connection can run on now, cheapest first (offers without a price first).
    fn offers(&self, values: &Values) -> Result<Vec<CloudOffer>>;

    /// For a provider that does not provision ([`CloudProvider::provisions`] false): the worker that already
    /// runs. `None` for providers that launch instances.
    fn fixed_endpoint(&self, values: &Values) -> Option<Endpoint> {
        let _ = values;
        None
    }

    /// Launches an instance that runs the worker ([`bootstrap::script`]) and returns the provider's id of it.
    /// [`ErrorKind::NoOffer`] for an unknown offer, [`ErrorKind::Refused`] when the provider has no capacity or
    /// the account no balance. After [`ErrorKind::Unavailable`] the instance may exist all the same (the request
    /// was sent and its answer lost): the caller looks for it with [`Provider::find`].
    fn launch(&self, values: &Values, spec: &LaunchSpec) -> Result<String> {
        let _ = (values, spec);
        Err(self.does_not_provision())
    }

    /// Where the instance `provider_ref` is.
    fn state(&self, values: &Values, provider_ref: &str) -> Result<InstanceState> {
        let _ = (values, provider_ref);
        Err(self.does_not_provision())
    }

    /// The provider's ids of the live instances named exactly `name` (normally none or one): how a launch whose
    /// answer was lost is found again.
    fn find(&self, values: &Values, name: &str) -> Result<Vec<String>> {
        let _ = (values, name);
        Err(self.does_not_provision())
    }

    /// Terminates the instance `provider_ref` for good. `Ok` once the provider no longer has it, also when it was
    /// gone before the call.
    fn terminate(&self, values: &Values, provider_ref: &str) -> Result<()> {
        let _ = (values, provider_ref);
        Err(self.does_not_provision())
    }

    /// The error of a provisioning call on a provider that does not provision.
    #[doc(hidden)]
    fn does_not_provision(&self) -> Error {
        Error::new(ErrorKind::Invalid, format!("{} does not launch instances", self.info().name))
    }
}

/// Every provider FOLO ships, in display order.
pub fn providers() -> Vec<Arc<dyn Provider>> {
    vec![Arc::new(runpod::RunPod::new()), Arc::new(remote::Remote::new())]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(pairs: &[(&str, &str)]) -> Values {
        Values::from(pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect::<BTreeMap<_, _>>())
    }

    #[test]
    fn values_trim_drop_empties_and_hide_themselves() {
        let v = values(&[("api_key", "  rpa_SECRET \n"), ("worker_base_url", "  ")]);
        assert_eq!(v.get("api_key"), Some("rpa_SECRET"));
        assert_eq!(v.get("worker_base_url"), None);
        let e = v.require("worker_base_url").unwrap_err();
        assert_eq!((e.kind, e.code()), (ErrorKind::Invalid, "invalid_body"));
        assert!(e.message.contains("worker_base_url"));
        let shown = format!("{v:?}");
        assert!(shown.contains("api_key") && !shown.contains("SECRET"), "{shown}");
    }

    #[test]
    fn worker_base_url_prefers_the_connection() {
        let own = values(&[(WORKER_BASE_URL_FIELD, "https://mirror.example/folo/")]);
        assert_eq!(worker_base_url(&own, "0.1.0"), "https://mirror.example/folo");
        // Without the field (and without the environment variable, which tests leave alone) it is the release.
        if std::env::var(WORKER_BASE_URL_ENV).is_err() {
            assert_eq!(
                worker_base_url(&Values::default(), "0.1.0"),
                "https://github.com/himent12/folo/releases/download/v0.1.0"
            );
        }
    }

    #[test]
    fn worker_download_tells_missing_from_unknown() {
        use crate::fixture::{Answer, Fixture};
        let ok = || Answer::json(200, "ok");
        let f = Fixture::serve(vec![ok(), ok()]);
        assert_eq!(worker_download(&f.url("/v0.1.0/")), WorkerDownload::Found);
        let paths: Vec<String> = f.requests().into_iter().map(|r| r.path).collect();
        assert_eq!(paths, ["/v0.1.0/install.sh", "/v0.1.0/folo-x86_64-unknown-linux-gnu.tar.gz.sha256"]);
        assert_eq!(WorkerDownload::Found.problem(), None);

        // The installer is there and the archive is not: a release without a Linux build.
        let f = Fixture::serve(vec![ok(), Answer::json(404, "{}")]);
        let missing = worker_download(&f.url("/v0.1.0"));
        let url = f.url("/v0.1.0/folo-x86_64-unknown-linux-gnu.tar.gz.sha256");
        assert_eq!(missing, WorkerDownload::Missing { url: url.clone(), status: 404 });
        let problem = missing.problem().unwrap();
        assert!(problem.contains(&format!("{url} answers 404")), "{problem}");
        assert!(
            problem.contains("worker download URL") && problem.contains(WORKER_BASE_URL_ENV),
            "{problem}"
        );

        // A server error, a refusal of the moment or no answer says nothing about what the instance will find.
        for status in [503, 408, 425, 429] {
            let f = Fixture::serve(vec![Answer::json(status, "{}")]);
            let unknown = worker_download(&f.url("/x"));
            assert!(
                matches!(&unknown, WorkerDownload::Unknown(m) if m.contains(&status.to_string())),
                "{status}: {unknown:?}"
            );
        }
        // Any other refusal is an answer: the file is not there for the instance either.
        for status in [401, 403, 410] {
            let f = Fixture::serve(vec![Answer::json(status, "{}")]);
            assert!(
                matches!(worker_download(&f.url("/x")), WorkerDownload::Missing { status: s, .. } if s == status)
            );
        }
        let unknown = worker_download(&Fixture::hang_up().url("/x"));
        assert!(
            matches!(&unknown, WorkerDownload::Unknown(m) if m.contains("does not answer")),
            "{unknown:?}"
        );
        assert_eq!(unknown.problem(), None);
        assert_eq!(Error::new(ErrorKind::NoWorker, "x").code(), "cloud_boot_failed");
    }

    #[test]
    fn providers_have_unique_ids_and_secret_fields_are_marked() {
        let all = providers();
        let ids: Vec<String> = all.iter().map(|p| p.info().id).collect();
        assert_eq!(ids, ["runpod", "remote"]);
        for p in &all {
            let info = p.info();
            assert!(!info.fields.is_empty(), "{}", info.id);
            for f in &info.fields {
                let secretish = f.key.contains("key") || f.key.contains("token");
                assert_eq!(f.secret, secretish, "{}.{}", info.id, f.key);
            }
        }
    }

    #[test]
    fn debug_of_secrets_is_redacted() {
        let e = Endpoint { url: "https://w".into(), token: "s3cret".into() };
        assert!(!format!("{e:?}").contains("s3cret"));
        let spec = LaunchSpec {
            name: "folo-run_1-ab".into(),
            offer_id: "o".into(),
            disk_gb: 40,
            token: "s3cret".into(),
            lifetime_s: 3600,
            worker_base_url: "https://x".into(),
        };
        assert!(!format!("{spec:?}").contains("s3cret"));
    }
}
