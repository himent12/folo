//! RunPod (docs/CLOUD.md, `runpod`): an on-demand GPU pod per run, rented through RunPod's REST API v2
//! (<https://docs.runpod.io/api-reference-v2>), and the account's email and balance through its GraphQL API (REST
//! v2 has no account endpoint).
//!
//! A pod runs the stock image [`IMAGE`] with the [`bootstrap`](crate::bootstrap) script as its command, the
//! worker's token in `FOLO_TOKEN` and `NVIDIA_DRIVER_CAPABILITIES=all` (so the NVIDIA runtime mounts the Vulkan
//! driver too), and exposes [`WORKER_PORT`](crate::WORKER_PORT) through RunPod's HTTPS proxy
//! (`https://<pod id>-7878.proxy.runpod.net`). It terminates itself with its pod-scoped key ([`SELF_TERMINATE`]).
//!
//! `FOLO_CLOUD_RUNPOD_IMAGE` ([`IMAGE_ENV`]) replaces the image, for experiments: any image with bash and apt-get
//! whose entrypoint can be replaced.
//!
//! Requests send exactly the documented fields (v2 rejects unknown ones with `422`); answers are read tolerantly
//! (unknown fields ignored, missing ones defaulted, numbers that arrive as strings accepted).

use std::collections::BTreeMap;

use folo_api::{CloudAvailability, CloudField, CloudOffer, CloudProvider};
use serde::{Deserialize, Deserializer, Serialize};

use crate::bootstrap;
use crate::http::{Api, Reply};
use crate::{
    Account, Error, ErrorKind, InstanceState, LaunchSpec, Provider, Result, Values, WORKER_BASE_URL_FIELD,
    WORKER_PORT,
};

/// RunPod's REST API v2.
pub const API_URL: &str = "https://api.runpod.io/v2";
/// RunPod's GraphQL API (the account's email and balance).
pub const GRAPHQL_URL: &str = "https://api.runpod.io/graphql";
/// Where a pod's HTTP port is served: `{id}` is the pod id, `{port}` the port inside the pod
/// (<https://docs.runpod.io/pods/configuration/expose-ports>).
pub const PROXY_URL: &str = "https://{id}-{port}.proxy.runpod.net";
/// The image a pod starts from.
pub const IMAGE: &str = "ubuntu:22.04";
/// Replaces [`IMAGE`], for experiments.
pub const IMAGE_ENV: &str = "FOLO_CLOUD_RUNPOD_IMAGE";

/// The body of the bootstrap's `self_terminate` for a pod: RunPod sets `RUNPOD_POD_ID` and a pod-scoped
/// `RUNPOD_API_KEY` in every pod; `runpodctl` (preinstalled in RunPod's own images, under two spellings across
/// versions) is the fallback (<https://docs.runpod.io/pods/manage-pods>).
pub const SELF_TERMINATE: &str = r#"  if [ -z "${RUNPOD_POD_ID:-}" ]; then
    log "error: RUNPOD_POD_ID is not set"
    return 1
  fi
  curl -fsS --max-time 30 --request DELETE --url "https://api.runpod.io/v2/pods/$RUNPOD_POD_ID" \
    --header "Authorization: Bearer ${RUNPOD_API_KEY:-}" \
    || runpodctl pod delete "$RUNPOD_POD_ID" \
    || runpodctl remove pod "$RUNPOD_POD_ID""#;

/// The connection field of the API key.
const API_KEY: &str = "api_key";
/// Pods asked for per page when listing them.
const PAGE_SIZE: u32 = 100;
/// Pages listed at most when looking for a pod by name (10 000 pods).
const MAX_PAGES: usize = 100;
/// Characters of RunPod's text kept in an error message.
const MAX_DETAIL_CHARS: usize = 300;

const REJECTED: &str = "RunPod rejected this API key";
const CANNOT_MANAGE_PODS: &str = "this RunPod API key cannot manage pods: give it read and write access to Pods \
                                  (RunPod console → Settings → API Keys)";

/// RunPod: GPU pods rented per run.
#[derive(Clone, Debug)]
pub struct RunPod {
    api: Api,
    rest: String,
    graphql: String,
    proxy: String,
    /// Whether the worker's download is asked for before a launch and at a check ([`crate::worker_download`]).
    probe: bool,
}

impl Default for RunPod {
    fn default() -> RunPod {
        RunPod::new()
    }
}

impl RunPod {
    /// The provider against its real service.
    pub fn new() -> RunPod {
        RunPod { probe: true, ..RunPod::with_urls(API_URL, GRAPHQL_URL, PROXY_URL) }
    }

    /// The provider against other URLs (a test fixture): the REST base (`.../v2`), the GraphQL endpoint and the
    /// proxy pattern ([`PROXY_URL`]). It does not ask for the worker's download (a fixture has none).
    pub fn with_urls(rest: &str, graphql: &str, proxy: &str) -> RunPod {
        RunPod {
            api: Api::new("RunPod"),
            rest: rest.trim_end_matches('/').to_string(),
            graphql: graphql.to_string(),
            proxy: proxy.to_string(),
            probe: false,
        }
    }

    /// What stands between an instance and its worker, if this machine can tell: the place it installs FOLO from
    /// answers that the files are not there.
    fn download_problem(&self, base_url: &str) -> Option<String> {
        self.probe.then(|| crate::worker_download(base_url).problem()).flatten()
    }

    /// The error of an answer the call has no meaning of its own for.
    fn failure(&self, reply: &Reply) -> Error {
        let problem = Problem::of(reply);
        let with = |what: &str| match problem.text() {
            Some(detail) => format!("{what}: {detail}"),
            None => what.to_string(),
        };
        match reply.status {
            401 => Error::new(ErrorKind::Auth, REJECTED),
            402 => Error::new(ErrorKind::Refused, with("RunPod refused: the account's balance is too low")),
            403 => Error::new(ErrorKind::Auth, CANNOT_MANAGE_PODS),
            400 | 409 => Error::new(ErrorKind::Refused, with("RunPod refused")),
            422 => {
                let mut message =
                    with("RunPod did not accept FOLO's request (a FOLO bug or a change of RunPod's API)");
                let errors = problem.errors();
                if !errors.is_empty() {
                    message = format!("{message} ({})", errors.join("; "));
                }
                Error::new(
                    ErrorKind::Unavailable,
                    folo_api::codes::cap(&message, folo_api::codes::MAX_MESSAGE_CHARS),
                )
            }
            429 => Error::new(
                ErrorKind::Unavailable,
                "RunPod is rate limiting this API key; try again in a minute",
            ),
            s => Error::new(ErrorKind::Unavailable, with(&format!("RunPod answered {s}"))),
        }
    }

    /// The account's email and balance, when GraphQL tells.
    fn account(&self, key: &str) -> Option<Account> {
        let query = serde_json::json!({ "query": "query { myself { email clientBalance } }" });
        let reply = self.api.post(&self.graphql, key, &query).ok().filter(Reply::is_success)?;
        let me = reply.json::<Graphql>().ok()?.data?.myself?;
        Some(Account {
            label: me.email.filter(|e| !e.is_empty()).map(|e| reply.scrub(&e)),
            balance_usd: me.client_balance,
            note: None,
        })
    }

    fn pod_url(&self, id: &str) -> String {
        format!("{}/pods/{id}", self.rest)
    }

    /// One page of the account's pods.
    fn pods(&self, key: &str, cursor: Option<&str>) -> Result<PodList> {
        let mut url = format!("{}/pods?limit={PAGE_SIZE}", self.rest);
        if let Some(c) = cursor {
            url = format!("{url}&cursor={}", folo_client::encode(c));
        }
        let reply = self.api.get(&url, key)?;
        if !reply.is_success() {
            return Err(self.failure(&reply));
        }
        // The ids are used as they came (looked up, terminated): never scrubbed.
        reply.json()
    }

    /// The offers of one cloud.
    fn catalog(&self, key: &str, cloud: Cloud) -> Result<Vec<CloudOffer>> {
        let url = format!(
            "{}/catalog/gpus?include=AVAILABILITY&product=POD&cloud={}&count=1",
            self.rest,
            cloud.wire()
        );
        let reply = self.api.get(&url, key)?;
        if !reply.is_success() {
            return Err(self.failure(&reply));
        }
        // A GPU's name is shown and logged ("launching <gpu>"): scrubbed. Its id goes back to RunPod in a launch:
        // as it came.
        let mut list: GpuList = reply.json()?;
        for g in &mut list.gpus {
            // (A GPU without a name is shown by its id.)
            g.name = reply.scrub(if g.name.is_empty() { &g.id } else { &g.name });
        }
        Ok(list.gpus.iter().filter_map(|g| offer(g, cloud)).collect())
    }
}

impl Provider for RunPod {
    fn info(&self) -> CloudProvider {
        CloudProvider {
            id: "runpod".into(),
            name: "RunPod".into(),
            about: "Rents an on-demand GPU pod on RunPod for each run and terminates it when the run ends.".into(),
            docs_url: Some("https://www.runpod.io/console/user/settings".into()),
            fields: vec![
                CloudField {
                    key: API_KEY.into(),
                    label: "API key".into(),
                    secret: true,
                    required: true,
                    placeholder: Some("rpa_…".into()),
                    help: Some(
                        "RunPod console → Settings → API Keys; the key needs read and write access to pods.".into(),
                    ),
                },
                CloudField {
                    key: WORKER_BASE_URL_FIELD.into(),
                    label: "Worker download URL".into(),
                    secret: false,
                    required: false,
                    placeholder: Some("https://example.com/folo".into()),
                    help: Some(
                        "Where instances download FOLO from (install.sh and the release archive) when not the \
                         public release of this version: a mirror, or your own build."
                            .into(),
                    ),
                },
            ],
            provisions: true,
            self_terminates: true,
        }
    }

    fn check(&self, values: &Values) -> Result<Account> {
        let key = api_key(values)?;
        if let Some(url) = values.get(WORKER_BASE_URL_FIELD) {
            bootstrap::check_base_url(url)?;
        }
        // Listing pods needs what FOLO needs: a key that works and may see pods.
        let reply = self.api.get(&format!("{}/pods?limit=1", self.rest), key)?;
        if !reply.is_success() {
            return Err(self.failure(&reply));
        }
        // Best effort: a working key is a working connection, also when GraphQL does not answer.
        let mut account = self.account(key).unwrap_or_default();
        // A connection whose instances cannot install FOLO works, and every run on it would fail: say so here.
        let base_url = crate::worker_base_url(values, env!("CARGO_PKG_VERSION"));
        account.note = self.download_problem(&base_url).map(|p| sentence(&p));
        Ok(account)
    }

    fn offers(&self, values: &Values) -> Result<Vec<CloudOffer>> {
        let key = api_key(values)?;
        let mut offers = self.catalog(key, Cloud::Secure)?;
        offers.extend(self.catalog(key, Cloud::Community)?);
        offers.sort_by(|a, b| {
            let price = |o: &CloudOffer| o.price_per_hour.unwrap_or(0.0);
            price(a).total_cmp(&price(b)).then_with(|| a.gpu.cmp(&b.gpu)).then_with(|| a.id.cmp(&b.id))
        });
        Ok(offers)
    }

    fn launch(&self, values: &Values, spec: &LaunchSpec) -> Result<String> {
        let key = api_key(values)?;
        let (gpu, cloud) = parse_offer(&spec.offer_id).ok_or_else(|| {
            Error::new(
                ErrorKind::NoOffer,
                format!(
                    "RunPod has no offer `{}` (offers are `<GPU type>@secure` or `<GPU type>@community`)",
                    spec.offer_id
                ),
            )
        })?;
        if spec.token.is_empty() {
            return Err(Error::new(ErrorKind::Invalid, "a worker needs a token"));
        }
        if spec.disk_gb == 0 {
            return Err(Error::new(ErrorKind::Invalid, "a pod needs at least 1 GB of disk"));
        }
        let script = bootstrap::script(&bootstrap::Spec {
            base_url: &spec.worker_base_url,
            port: WORKER_PORT,
            lifetime_s: spec.lifetime_s,
            terminate: SELF_TERMINATE,
        })?;
        // Before anything is rented: a pod that cannot download FOLO only costs money until it gives up.
        if let Some(problem) = self.download_problem(&spec.worker_base_url) {
            return Err(Error::new(ErrorKind::NoWorker, problem));
        }
        let image = std::env::var(IMAGE_ENV).ok().filter(|i| !i.trim().is_empty());
        let port = format!("{WORKER_PORT}/http");
        let body = CreatePod {
            name: &spec.name,
            image: image.as_deref().unwrap_or(IMAGE).trim(),
            gpu: GpuRequest { id: gpu, count: 1 },
            cloud: cloud.wire(),
            disk: spec.disk_gb,
            ports: [&port],
            env: BTreeMap::from([("FOLO_TOKEN", spec.token.as_str()), ("NVIDIA_DRIVER_CAPABILITIES", "all")]),
            entrypoint: ["/bin/bash", "-c"],
            cmd: [&script],
        };
        // The worker's token is in the request too, and a refusal may quote what it refused.
        let reply = self.api.post(&format!("{}/pods", self.rest), key, &body)?.without(&spec.token);
        match reply.status {
            200..=299 => {
                let pod: Pod = reply.json()?;
                // The provider's reference, used as it came.
                check_pod_id(&pod.id).map_err(|_| {
                    Error::new(ErrorKind::Unavailable, "RunPod created a pod but did not answer with its id")
                })?;
                Ok(pod.id)
            }
            // An account that may not use this GPU's pool, or a key scoped away from it: RunPod's text says which.
            403 => Err(Error::new(
                ErrorKind::Refused,
                match Problem::of(&reply).text() {
                    Some(detail) => format!("RunPod refused the pod: {detail}"),
                    None => "RunPod refused the pod".into(),
                },
            )),
            404 => Err(Error::new(ErrorKind::NoOffer, format!("RunPod does not offer the GPU type `{gpu}`"))),
            _ => Err(self.failure(&reply)),
        }
    }

    fn state(&self, values: &Values, provider_ref: &str) -> Result<InstanceState> {
        let key = api_key(values)?;
        check_pod_id(provider_ref)?;
        let reply = self.api.get(&self.pod_url(provider_ref), key)?;
        // What a deleted pod answers is not documented: `404` and `TERMINATED` both mean gone.
        if reply.status == 404 {
            return Ok(InstanceState::Gone);
        }
        if !reply.is_success() {
            return Err(self.failure(&reply));
        }
        let pod: Pod = reply.json()?;
        match pod.status.as_str() {
            "PROVISIONING" | "STARTING" => Ok(InstanceState::Starting),
            "RUNNING" => Ok(InstanceState::Running {
                url: self.proxy.replace("{id}", provider_ref).replace("{port}", &WORKER_PORT.to_string()),
            }),
            "EXITED" | "ERROR" => Ok(InstanceState::Stopped),
            "TERMINATED" => Ok(InstanceState::Gone),
            // Neither `Starting` (it may never start) nor `Gone` (it may be billed): the caller asks again.
            other => Err(Error::new(
                ErrorKind::Unavailable,
                format!(
                    "RunPod reports pod {provider_ref} as `{}`, a status FOLO does not know",
                    reply.scrub(other)
                ),
            )),
        }
    }

    fn find(&self, values: &Values, name: &str) -> Result<Vec<String>> {
        let key = api_key(values)?;
        // The API has no filter by name: list every pod and look.
        let mut ids = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let page = self.pods(key, cursor.as_deref())?;
            ids.extend(
                page.pods.into_iter().filter(|p| p.name == name && p.status != "TERMINATED").map(|p| p.id),
            );
            match page.pagination.next_cursor.filter(|c| !c.is_empty()) {
                Some(next) if cursor.as_ref() != Some(&next) => cursor = Some(next),
                _ => return Ok(ids),
            }
        }
        Err(Error::new(ErrorKind::Unavailable, format!("RunPod listed more than {MAX_PAGES} pages of pods")))
    }

    fn terminate(&self, values: &Values, provider_ref: &str) -> Result<()> {
        let key = api_key(values)?;
        check_pod_id(provider_ref)?;
        let reply = self.api.delete(&self.pod_url(provider_ref), key)?;
        match reply.status {
            200..=299 | 404 => Ok(()),
            _ => Err(self.failure(&reply)),
        }
    }
}

/// `message` as a sentence of its own: a capital first, a full stop last.
fn sentence(message: &str) -> String {
    let mut chars = message.chars();
    let first = chars.next().map(|c| c.to_uppercase().collect::<String>()).unwrap_or_default();
    format!("{first}{}.", chars.as_str().trim_end_matches('.'))
}

/// The connection's API key, which must look like one (it goes into a header).
fn api_key(values: &Values) -> Result<&str> {
    let key = values.require(API_KEY)?;
    if key.bytes().all(|b| b.is_ascii_graphic()) {
        Ok(key)
    } else {
        Err(Error::new(
            ErrorKind::Invalid,
            "`api_key` is not a RunPod API key: it contains spaces or control characters",
        ))
    }
}

/// A pod id goes into URLs and a host name: letters, digits, `-` and `_` only.
fn check_pod_id(id: &str) -> Result<()> {
    if !id.is_empty() && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_') {
        Ok(())
    } else {
        Err(Error::new(ErrorKind::Invalid, format!("`{id}` is not a RunPod pod id")))
    }
}

/// A RunPod cloud.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cloud {
    Secure,
    Community,
}

impl Cloud {
    /// The API's spelling.
    fn wire(self) -> &'static str {
        match self {
            Cloud::Secure => "SECURE",
            Cloud::Community => "COMMUNITY",
        }
    }

    /// The suffix of an offer id.
    fn suffix(self) -> &'static str {
        match self {
            Cloud::Secure => "secure",
            Cloud::Community => "community",
        }
    }

    /// The offer's detail.
    fn label(self) -> &'static str {
        match self {
            Cloud::Secure => "Secure Cloud",
            Cloud::Community => "Community Cloud",
        }
    }
}

/// `<gpu type id>@secure|community` → its parts.
fn parse_offer(id: &str) -> Option<(&str, Cloud)> {
    let (gpu, cloud) = id.rsplit_once('@')?;
    let cloud = [Cloud::Secure, Cloud::Community].into_iter().find(|c| c.suffix() == cloud)?;
    (!gpu.trim().is_empty()).then_some((gpu, cloud))
}

/// The offer of `gpu` on `cloud`, when it is offered there with a price.
fn offer(gpu: &GpuType, cloud: Cloud) -> Option<CloudOffer> {
    // Only NVIDIA: the worker finds its GPU through the NVIDIA container runtime's Vulkan driver (bootstrap).
    if gpu.manufacturer != "NVIDIA" || gpu.id.is_empty() {
        return None;
    }
    let (offered, price) = match cloud {
        Cloud::Secure => (gpu.secure, gpu.price.secure),
        Cloud::Community => (gpu.community, gpu.price.community),
    };
    if !offered {
        return None;
    }
    let price = price.filter(|p| p.is_finite() && *p > 0.0)?;
    let availability = match gpu.availability.as_deref() {
        Some("HIGH") => CloudAvailability::High,
        Some("MEDIUM") => CloudAvailability::Medium,
        Some("LOW") => CloudAvailability::Low,
        Some("NONE") => CloudAvailability::None,
        _ => CloudAvailability::Unknown,
    };
    let gib = gpu.memory.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(0.0);
    Some(CloudOffer {
        id: format!("{}@{}", gpu.id, cloud.suffix()),
        gpu: if gpu.name.is_empty() { gpu.id.clone() } else { gpu.name.clone() },
        gpu_count: 1,
        vram_bytes: (gib * (1u64 << 30) as f64) as u64,
        price_per_hour: Some(price),
        region: None,
        detail: Some(cloud.label().into()),
        availability,
    })
}

/// `POST /v2/pods` body: exactly the documented fields FOLO sets
/// (<https://docs.runpod.io/api-reference-v2/pods/create-a-pod>).
#[derive(Serialize)]
struct CreatePod<'a> {
    name: &'a str,
    image: &'a str,
    gpu: GpuRequest<'a>,
    cloud: &'static str,
    disk: u32,
    ports: [&'a str; 1],
    env: BTreeMap<&'static str, &'a str>,
    entrypoint: [&'static str; 2],
    cmd: [&'a str; 1],
}

/// [`CreatePod::gpu`].
#[derive(Serialize)]
struct GpuRequest<'a> {
    id: &'a str,
    count: u32,
}

/// A pod, as far as FOLO reads it.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Pod {
    id: String,
    name: String,
    status: String,
}

/// `GET /v2/pods`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct PodList {
    pods: Vec<Pod>,
    pagination: Pagination,
}

/// [`PodList::pagination`].
#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Pagination {
    next_cursor: Option<String>,
}

/// `GET /v2/catalog/gpus`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GpuList {
    gpus: Vec<GpuType>,
}

/// A GPU type of the catalog (<https://docs.runpod.io/api-reference-v2/catalog/list-gpu-types>).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GpuType {
    id: String,
    name: String,
    manufacturer: String,
    /// VRAM in GB (GiB).
    #[serde(deserialize_with = "number")]
    memory: Option<f64>,
    secure: bool,
    community: bool,
    price: Prices,
    /// For the cloud asked.
    availability: Option<String>,
}

/// [`GpuType::price`]: US dollars per hour for one GPU.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Prices {
    #[serde(deserialize_with = "number")]
    secure: Option<f64>,
    #[serde(deserialize_with = "number")]
    community: Option<f64>,
}

/// An error answer (`application/problem+json`).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Problem {
    title: Option<String>,
    detail: Option<String>,
    errors: Option<Vec<serde_json::Value>>,
    /// The body when it is not a problem document (a proxy's page).
    #[serde(skip)]
    raw: Option<String>,
}

impl Problem {
    /// The problem of `reply`, every text in it scrubbed of the request's credentials ([`Reply::scrub`]): they are
    /// decoded here, so an escaped one is spelled out.
    fn of(reply: &Reply) -> Problem {
        let p: Problem = serde_json::from_slice(&reply.body).unwrap_or_else(|_| Problem {
            raw: Some(String::from_utf8_lossy(&reply.body).into_owned()),
            ..Problem::default()
        });
        let scrub = |t: Option<String>| t.map(|t| reply.scrub(&t));
        // Whole, before anything shortens it: a credential cut in two would not be found.
        let whole = |v: &serde_json::Value| match v {
            serde_json::Value::String(s) => reply.scrub(s),
            other => reply.scrub(&other.to_string()),
        };
        Problem {
            title: scrub(p.title),
            detail: scrub(p.detail),
            errors: p.errors.map(|e| e.iter().map(|v| serde_json::Value::String(whole(v))).collect()),
            raw: scrub(p.raw),
        }
    }

    /// What RunPod said, shortened: the detail, else the title, else the start of the body.
    fn text(&self) -> Option<String> {
        [&self.detail, &self.title, &self.raw]
            .into_iter()
            .flatten()
            .map(|t| t.trim())
            .find(|t| !t.is_empty())
            .map(|t| folo_api::codes::cap(t, MAX_DETAIL_CHARS))
    }

    /// The validation errors of a `422`, one per entry.
    fn errors(&self) -> Vec<String> {
        self.errors.iter().flatten().map(text_of).collect()
    }
}

/// One entry of [`Problem::errors`] as text (documented as strings; anything else as JSON).
fn text_of(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::String(s) => folo_api::codes::cap(s, MAX_DETAIL_CHARS),
        other => folo_api::codes::cap(&other.to_string(), MAX_DETAIL_CHARS),
    }
}

/// The GraphQL answer to `myself`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct Graphql {
    data: Option<GraphqlData>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct GraphqlData {
    myself: Option<Myself>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Myself {
    email: Option<String>,
    #[serde(deserialize_with = "number")]
    client_balance: Option<f64>,
}

/// A number that may arrive as a JSON number, a numeric string or `null`.
fn number<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Option<f64>, D::Error> {
    Ok(match serde_json::Value::deserialize(d)? {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.trim().parse().ok(),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture::{Answer, Fixture, bash_syntax};

    const KEY: &str = "rpa_TESTKEY0123456789abcdef";
    const TOKEN: &str = "worker-token-0123456789";
    const NO_PODS: &str = r#"{"pods":[],"pagination":{"nextCursor":null,"hasNextPage":false}}"#;

    fn values() -> Values {
        Values::from(BTreeMap::from([(API_KEY.to_string(), KEY.to_string())]))
    }

    fn runpod(f: &Fixture) -> RunPod {
        RunPod::with_urls(&f.url("/v2"), &f.url("/graphql"), &format!("{}/proxy/{{id}}-{{port}}", f.base()))
    }

    fn spec(offer_id: &str) -> LaunchSpec {
        LaunchSpec {
            name: "folo-run_7d1e-4c1a".into(),
            offer_id: offer_id.into(),
            disk_gb: 40,
            token: TOKEN.into(),
            lifetime_s: 3600,
            worker_base_url: "https://github.com/himent12/folo/releases/download/v0.1.0".into(),
        }
    }

    fn problem(status: u16, detail: &str) -> Answer {
        let body = serde_json::json!({ "title": "Error", "status": status, "detail": detail });
        Answer::json(status, body.to_string())
    }

    /// Every request carried the key as a bearer header and never in its URL.
    fn assert_key_only_in_headers(f: &Fixture) {
        for r in f.requests() {
            assert!(!r.path.contains(KEY), "{}", r.path);
            assert_eq!(r.header("authorization"), Some(format!("Bearer {KEY}").as_str()), "{}", r.path);
        }
    }

    /// The error has `kind`, mentions `part` and shows neither the key nor the token.
    fn assert_error(e: &Error, kind: ErrorKind, part: &str) {
        assert_eq!(e.kind, kind, "{e:?}");
        assert!(e.message.contains(part), "`{part}` not in: {}", e.message);
        for shown in [e.message.clone(), format!("{e:?}")] {
            assert!(!shown.contains(KEY) && !shown.contains(TOKEN), "{shown}");
        }
    }

    #[test]
    fn info_describes_the_connection() {
        let info = RunPod::new().info();
        assert_eq!((info.id.as_str(), info.provisions, info.self_terminates), ("runpod", true, true));
        let fields: Vec<(&str, bool, bool)> =
            info.fields.iter().map(|f| (f.key.as_str(), f.secret, f.required)).collect();
        assert_eq!(fields, [("api_key", true, true), ("worker_base_url", false, false)]);
        assert!(info.fields[0].help.as_deref().unwrap().contains("write access"));
    }

    #[test]
    fn check_reads_the_account() {
        let f = Fixture::serve(vec![
            Answer::json(200, NO_PODS),
            Answer::json(
                200,
                r#"{"data":{"myself":{"id":"u1","email":"you@example.com","clientBalance":"42.17"}}}"#,
            ),
        ]);
        let account = runpod(&f).check(&values()).unwrap();
        assert_eq!(
            account,
            Account { label: Some("you@example.com".into()), balance_usd: Some(42.17), note: None }
        );
        let r = f.requests();
        assert_eq!((r[0].method.as_str(), r[0].path.as_str()), ("GET", "/v2/pods?limit=1"));
        assert_eq!((r[1].method.as_str(), r[1].path.as_str()), ("POST", "/graphql"));
        assert!(r[1].json()["query"].as_str().unwrap().contains("myself"));
        assert_key_only_in_headers(&f);
    }

    #[test]
    fn check_notes_a_worker_download_that_is_not_there() {
        let pods = || Answer::json(200, NO_PODS);
        let me = || Answer::json(200, r#"{"data":{"myself":{"email":"you@example.com","clientBalance":5}}}"#);
        // Pods, the account, then the two files of the download: the installer is there, the archive is not.
        let f = Fixture::serve(vec![pods(), me(), Answer::json(200, "ok"), Answer::json(404, "{}")]);
        let mut map = values().as_map().clone();
        map.insert(WORKER_BASE_URL_FIELD.into(), f.url("/mirror"));
        let probing = RunPod { probe: true, ..runpod(&f) };
        let account = probing.check(&Values::from(map.clone())).unwrap();
        assert_eq!(account.label.as_deref(), Some("you@example.com"));
        let note = account.note.expect("a note");
        assert!(note.starts_with("Instances cannot install FOLO: ") && note.ends_with("_URL."), "{note}");
        assert!(note.contains("/mirror/folo-x86_64-unknown-linux-gnu.tar.gz.sha256 answers 404"), "{note}");
        let r = f.requests();
        assert_eq!(r.len(), 4);
        // The key goes to RunPod only, never to the place the worker is downloaded from.
        assert!(r[2..].iter().all(|r| r.header("authorization").is_none() && r.path.starts_with("/mirror/")));

        // A download that is there leaves no note.
        let f = Fixture::serve(vec![pods(), me(), Answer::json(200, "ok"), Answer::json(200, "ok")]);
        map.insert(WORKER_BASE_URL_FIELD.into(), f.url("/mirror"));
        let probing = RunPod { probe: true, ..runpod(&f) };
        assert_eq!(probing.check(&Values::from(map)).unwrap().note, None);
    }

    #[test]
    fn launch_rents_nothing_when_the_worker_cannot_be_downloaded() {
        let f = Fixture::serve(vec![Answer::json(404, "{}")]);
        let probing = RunPod { probe: true, ..runpod(&f) };
        let missing = LaunchSpec { worker_base_url: f.url("/v9.9.9"), ..spec("NVIDIA L4@secure") };
        let e = probing.launch(&values(), &missing).unwrap_err();
        assert_error(&e, ErrorKind::NoWorker, "/v9.9.9/install.sh answers 404");
        assert_eq!(e.code(), "cloud_boot_failed");
        let r = f.requests();
        assert_eq!((r.len(), r[0].method.as_str()), (1, "GET"), "no pod was asked for");

        // With the download there, the pod is created.
        let created = Answer::json(201, r#"{"id":"abc123xyz","status":"PROVISIONING"}"#);
        let f = Fixture::serve(vec![Answer::json(200, "ok"), Answer::json(200, "ok"), created]);
        let probing = RunPod { probe: true, ..runpod(&f) };
        let there = LaunchSpec { worker_base_url: f.url("/v0.1.0"), ..spec("NVIDIA L4@secure") };
        assert_eq!(probing.launch(&values(), &there).unwrap(), "abc123xyz");
        assert_eq!(f.requests()[2].path, "/v2/pods");
    }

    #[test]
    fn check_maps_a_rejected_key_and_one_without_pod_access() {
        let f = Fixture::serve(vec![problem(401, "invalid token")]);
        assert_error(
            &runpod(&f).check(&values()).unwrap_err(),
            ErrorKind::Auth,
            "RunPod rejected this API key",
        );
        assert_eq!(f.requests().len(), 1, "no GraphQL call after a rejected key");
        let f = Fixture::serve(vec![problem(403, "forbidden")]);
        assert_error(&runpod(&f).check(&values()).unwrap_err(), ErrorKind::Auth, "cannot manage pods");
    }

    #[test]
    fn check_passes_without_graphql() {
        for graphql in [
            problem(503, "down"),
            Answer::json(200, r#"{"data":{"myself":null},"errors":[{"message":"unauthorized"}]}"#),
            Answer::json(200, "not json"),
        ] {
            let f = Fixture::serve(vec![Answer::json(200, NO_PODS), graphql]);
            assert_eq!(runpod(&f).check(&values()).unwrap(), Account::default());
        }
    }

    #[test]
    fn check_refuses_malformed_values_before_any_request() {
        let f = Fixture::serve(Vec::new());
        let p = runpod(&f);
        assert_error(&p.check(&Values::default()).unwrap_err(), ErrorKind::Invalid, "api_key");
        let spaced = Values::from(BTreeMap::from([(API_KEY.to_string(), "rpa_a b".to_string())]));
        assert_error(&p.check(&spaced).unwrap_err(), ErrorKind::Invalid, "api_key");
        let mut map = values().as_map().clone();
        map.insert(WORKER_BASE_URL_FIELD.into(), "ftp://mirror".into());
        assert_error(&p.check(&Values::from(map)).unwrap_err(), ErrorKind::Invalid, "worker_base_url");
        assert!(f.requests().is_empty());
    }

    #[test]
    fn offers_merge_both_clouds_cheapest_first() {
        let secure = r#"{"gpus":[
            {"id":"NVIDIA GeForce RTX 4090","name":"RTX 4090","manufacturer":"NVIDIA","memory":24,"secure":true,
             "community":true,"price":{"secure":0.69,"community":0.34},"availability":"HIGH","extra":1},
            {"id":"NVIDIA A100 80GB PCIe","name":"A100 PCIe","manufacturer":"NVIDIA","memory":"80","secure":true,
             "community":false,"price":{"secure":"1.64","community":null},"availability":"LOW"},
            {"id":"AMD Instinct MI300X OAM","name":"MI300X","manufacturer":"AMD","memory":192,"secure":true,
             "community":false,"price":{"secure":2.49,"community":0}},
            {"id":"NVIDIA H100 80GB HBM3","name":"H100 SXM","manufacturer":"NVIDIA","memory":80,"secure":true,
             "community":false,"price":{"secure":2.99,"community":0}},
            {"id":"NVIDIA L4","name":"L4","manufacturer":"NVIDIA","memory":24,"secure":false,"community":true,
             "price":{"secure":0.5,"community":0.29},"availability":"NONE"}]}"#;
        let community = r#"{"gpus":[
            {"id":"NVIDIA GeForce RTX 4090","name":"RTX 4090","manufacturer":"NVIDIA","memory":24,"secure":true,
             "community":true,"price":{"secure":0.69,"community":0.34},"availability":"MEDIUM"},
            {"id":"NVIDIA A100 80GB PCIe","name":"A100 PCIe","manufacturer":"NVIDIA","memory":80,"secure":true,
             "community":false,"price":{"secure":1.64,"community":null},"availability":"NONE"},
            {"id":"AMD Instinct MI300X OAM","name":"MI300X","manufacturer":"AMD","memory":192,"secure":false,
             "community":true,"price":{"secure":2.49,"community":1.99},"availability":"HIGH"},
            {"id":"NVIDIA L4","name":"L4","manufacturer":"NVIDIA","memory":24,"secure":false,"community":true,
             "price":{"secure":0.5,"community":0.29},"availability":"NONE"}]}"#;
        let f = Fixture::serve(vec![Answer::json(200, secure), Answer::json(200, community)]);
        let offers = runpod(&f).offers(&values()).unwrap();
        let shown: Vec<(&str, &str, f64, CloudAvailability)> = offers
            .iter()
            .map(|o| (o.id.as_str(), o.detail.as_deref().unwrap(), o.price_per_hour.unwrap(), o.availability))
            .collect();
        assert_eq!(
            shown,
            [
                ("NVIDIA L4@community", "Community Cloud", 0.29, CloudAvailability::None),
                ("NVIDIA GeForce RTX 4090@community", "Community Cloud", 0.34, CloudAvailability::Medium),
                ("NVIDIA GeForce RTX 4090@secure", "Secure Cloud", 0.69, CloudAvailability::High),
                ("NVIDIA A100 80GB PCIe@secure", "Secure Cloud", 1.64, CloudAvailability::Low),
                ("NVIDIA H100 80GB HBM3@secure", "Secure Cloud", 2.99, CloudAvailability::Unknown),
            ]
        );
        let rtx = &offers[2];
        assert_eq!((rtx.gpu.as_str(), rtx.gpu_count, rtx.vram_bytes), ("RTX 4090", 1, 24 << 30));
        assert_eq!(offers[3].vram_bytes, 80 << 30);
        let paths: Vec<String> = f.requests().into_iter().map(|r| r.path).collect();
        assert_eq!(
            paths,
            [
                "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&count=1",
                "/v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=COMMUNITY&count=1",
            ]
        );
        assert_key_only_in_headers(&f);
    }

    #[test]
    fn offers_fail_as_their_requests_do() {
        let f = Fixture::serve(vec![problem(401, "nope")]);
        assert_error(&runpod(&f).offers(&values()).unwrap_err(), ErrorKind::Auth, "rejected");
        let f = Fixture::serve(vec![Answer::json(200, r#"{"gpus":[]}"#), problem(500, "boom")]);
        assert_error(
            &runpod(&f).offers(&values()).unwrap_err(),
            ErrorKind::Unavailable,
            "RunPod answered 500: boom",
        );
    }

    #[test]
    fn launch_sends_exactly_the_documented_fields() {
        let created = r#"{"id":"abc123xyz","name":"folo-run_7d1e-4c1a","status":"PROVISIONING","cost":0.34}"#;
        let f = Fixture::serve(vec![Answer::json(201, created)]);
        let spec = spec("NVIDIA GeForce RTX 4090@community");
        assert_eq!(runpod(&f).launch(&values(), &spec).unwrap(), "abc123xyz");
        let r = &f.requests()[0];
        assert_eq!((r.method.as_str(), r.path.as_str()), ("POST", "/v2/pods"));
        let body = r.json();
        let keys: Vec<&str> = body.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys, ["name", "image", "gpu", "cloud", "disk", "ports", "env", "entrypoint", "cmd"]);
        let image = std::env::var(IMAGE_ENV).ok().filter(|i| !i.trim().is_empty());
        assert_eq!(body["image"], image.as_deref().unwrap_or(IMAGE).trim());
        assert_eq!(body["name"], "folo-run_7d1e-4c1a");
        assert_eq!(body["gpu"], serde_json::json!({ "id": "NVIDIA GeForce RTX 4090", "count": 1 }));
        assert_eq!((&body["cloud"], &body["disk"]), (&"COMMUNITY".into(), &40.into()));
        assert_eq!(body["ports"], serde_json::json!(["7878/http"]));
        assert_eq!(
            body["env"],
            serde_json::json!({ "FOLO_TOKEN": TOKEN, "NVIDIA_DRIVER_CAPABILITIES": "all" })
        );
        assert_eq!(body["entrypoint"], serde_json::json!(["/bin/bash", "-c"]));
        let script = bootstrap::script(&bootstrap::Spec {
            base_url: &spec.worker_base_url,
            port: WORKER_PORT,
            lifetime_s: 3600,
            terminate: SELF_TERMINATE,
        })
        .unwrap();
        assert_eq!(body["cmd"], serde_json::json!([script]));
        // The token is in `env` and nowhere else.
        let mut rest = body.clone();
        rest.as_object_mut().unwrap().remove("env");
        assert!(!rest.to_string().contains(TOKEN));
        assert_key_only_in_headers(&f);
    }

    #[test]
    fn the_pod_script_parses() {
        let s = bootstrap::script(&bootstrap::Spec {
            base_url: "https://example.com/folo",
            port: WORKER_PORT,
            lifetime_s: 3600,
            terminate: SELF_TERMINATE,
        })
        .unwrap();
        assert!(s.contains("$RUNPOD_POD_ID") && s.contains("${RUNPOD_API_KEY:-}"));
        match bash_syntax(&s) {
            None => eprintln!("skipped: no bash on PATH"),
            Some(r) => assert_eq!(r, Ok(()), "{s}"),
        }
    }

    #[test]
    fn launch_maps_each_refusal() {
        let no_capacity = "There are no longer any instances available with the requested specifications.";
        let invalid = r#"{"title":"Unprocessable Entity","status":422,"detail":"validation failed",
                          "errors":["$: additional properties 'bogus' not allowed"]}"#;
        let busy = problem(429, "slow down").header("Retry-After", "0");
        let cases = [
            (vec![problem(400, no_capacity)], ErrorKind::Refused, "no longer any instances available"),
            (vec![problem(401, "invalid")], ErrorKind::Auth, "RunPod rejected this API key"),
            (vec![problem(402, "Insufficient balance.")], ErrorKind::Refused, "balance is too low"),
            (vec![problem(403, "no access to this pool")], ErrorKind::Refused, "no access to this pool"),
            (vec![problem(404, "resource not found")], ErrorKind::NoOffer, "`NVIDIA L4`"),
            (
                vec![Answer::json(422, invalid)],
                ErrorKind::Unavailable,
                "validation failed ($: additional properties 'bogus' not allowed)",
            ),
            (vec![problem(500, "internal")], ErrorKind::Unavailable, "RunPod answered 500: internal"),
            (vec![Answer::empty(502)], ErrorKind::Unavailable, "RunPod answered 502"),
            (
                vec![Answer::json(503, r#"{"title":"Service Unavailable","detail":null,"errors":null}"#)],
                ErrorKind::Unavailable,
                "RunPod answered 503: Service Unavailable",
            ),
            (vec![busy.clone(), busy.clone(), busy], ErrorKind::Unavailable, "rate limiting"),
        ];
        for (answers, kind, part) in cases {
            let f = Fixture::serve(answers);
            let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
            assert_error(&e, kind, part);
            assert_key_only_in_headers(&f);
        }
    }

    #[test]
    fn launch_refuses_what_it_cannot_send() {
        let f = Fixture::serve(Vec::new());
        let p = runpod(&f);
        for offer in ["NVIDIA L4", "NVIDIA L4@spot", "@secure", ""] {
            assert_error(&p.launch(&values(), &spec(offer)).unwrap_err(), ErrorKind::NoOffer, "has no offer");
        }
        let no_token = LaunchSpec { token: String::new(), ..spec("NVIDIA L4@secure") };
        assert_error(&p.launch(&values(), &no_token).unwrap_err(), ErrorKind::Invalid, "token");
        let no_disk = LaunchSpec { disk_gb: 0, ..spec("NVIDIA L4@secure") };
        assert_error(&p.launch(&values(), &no_disk).unwrap_err(), ErrorKind::Invalid, "disk");
        let bad_url = LaunchSpec { worker_base_url: "mirror.example".into(), ..spec("NVIDIA L4@secure") };
        assert_error(&p.launch(&values(), &bad_url).unwrap_err(), ErrorKind::Invalid, "worker_base_url");
        assert!(f.requests().is_empty());
    }

    #[test]
    fn a_created_pod_without_an_id_is_unavailable() {
        let f = Fixture::serve(vec![Answer::json(201, r#"{"status":"PROVISIONING"}"#)]);
        let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, ErrorKind::Unavailable, "did not answer with its id");
    }

    #[test]
    fn state_maps_every_status() {
        let pod = |status: &str| {
            Answer::json(200, format!(r#"{{"id":"abc123xyz","status":"{status}","runtime":null}}"#))
        };
        let f = Fixture::serve(vec![
            pod("PROVISIONING"),
            pod("STARTING"),
            pod("RUNNING"),
            pod("EXITED"),
            pod("ERROR"),
            pod("TERMINATED"),
            problem(404, "resource not found"),
        ]);
        let p = runpod(&f);
        let running = InstanceState::Running { url: format!("{}/proxy/abc123xyz-7878", f.base()) };
        for want in [
            InstanceState::Starting,
            InstanceState::Starting,
            running,
            InstanceState::Stopped,
            InstanceState::Stopped,
            InstanceState::Gone,
            InstanceState::Gone,
        ] {
            assert_eq!(p.state(&values(), "abc123xyz").unwrap(), want);
        }
        assert!(f.requests().iter().all(|r| r.method == "GET" && r.path == "/v2/pods/abc123xyz"));
        assert_key_only_in_headers(&f);
    }

    #[test]
    fn the_real_proxy_url_names_the_pod_and_port() {
        let f = Fixture::serve(vec![Answer::json(200, r#"{"id":"abc123xyz","status":"RUNNING"}"#)]);
        let p = RunPod::with_urls(&f.url("/v2"), &f.url("/graphql"), PROXY_URL);
        let url = "https://abc123xyz-7878.proxy.runpod.net".to_string();
        assert_eq!(p.state(&values(), "abc123xyz").unwrap(), InstanceState::Running { url });
    }

    #[test]
    fn state_does_not_guess_an_unknown_status() {
        let f = Fixture::serve(vec![Answer::json(200, r#"{"id":"abc123xyz","status":"PAUSED"}"#)]);
        assert_error(
            &runpod(&f).state(&values(), "abc123xyz").unwrap_err(),
            ErrorKind::Unavailable,
            "`PAUSED`",
        );
        assert_error(&runpod(&f).state(&values(), "../pods").unwrap_err(), ErrorKind::Invalid, "pod id");
        assert_eq!(f.requests().len(), 1);
    }

    #[test]
    fn find_follows_pages_and_skips_terminated_namesakes() {
        let f = Fixture::serve(vec![
            Answer::json(
                200,
                r#"{"pods":[{"id":"a1","name":"folo-x","status":"RUNNING"},
                {"id":"b2","name":"folo-y","status":"RUNNING"}],"pagination":{"nextCursor":"c 2","hasNextPage":true}}"#,
            ),
            Answer::json(
                200,
                r#"{"pods":[{"id":"c3","name":"folo-x","status":"TERMINATED"},
                {"id":"d4","name":"folo-x","status":"EXITED"}],"pagination":{"nextCursor":null,"hasNextPage":false}}"#,
            ),
        ]);
        assert_eq!(runpod(&f).find(&values(), "folo-x").unwrap(), ["a1", "d4"]);
        let paths: Vec<String> = f.requests().into_iter().map(|r| r.path).collect();
        assert_eq!(paths, ["/v2/pods?limit=100", "/v2/pods?limit=100&cursor=c%202"]);
        assert_key_only_in_headers(&f);
    }

    #[test]
    fn terminate_is_ok_once_the_pod_is_gone() {
        let f = Fixture::serve(vec![
            Answer::empty(204),
            problem(404, "resource not found"),
            problem(500, "oops"),
        ]);
        let p = runpod(&f);
        p.terminate(&values(), "abc123xyz").unwrap();
        p.terminate(&values(), "abc123xyz").unwrap();
        assert_error(&p.terminate(&values(), "abc123xyz").unwrap_err(), ErrorKind::Unavailable, "500");
        assert!(f.requests().iter().all(|r| r.method == "DELETE" && r.path == "/v2/pods/abc123xyz"));
        assert_key_only_in_headers(&f);
    }

    #[test]
    fn a_key_echoed_by_runpod_is_not_repeated() {
        let f = Fixture::serve(vec![problem(400, &format!("bad key {KEY}"))]);
        let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, ErrorKind::Refused, "bad key [redacted]");
    }

    #[test]
    fn a_worker_token_echoed_by_runpod_is_not_repeated() {
        // A validation error names the value it refused, and `env` carries the worker's token.
        let invalid = format!(
            r#"{{"title":"Unprocessable Entity","status":422,"detail":"validation failed","errors":[
                {{"message":"unexpected property","location":"body.env",
                  "value":{{"FOLO_TOKEN":"{TOKEN}","NVIDIA_DRIVER_CAPABILITIES":"all"}}}}]}}"#
        );
        let f = Fixture::serve(vec![Answer::json(422, invalid)]);
        let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, ErrorKind::Unavailable, r#""FOLO_TOKEN":"[redacted]""#);
        let f = Fixture::serve(vec![problem(400, &format!("bad env FOLO_TOKEN={TOKEN}"))]);
        let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert_error(&e, ErrorKind::Refused, "bad env FOLO_TOKEN=[redacted]");
    }

    /// `s` with every character as a JSON `\uXXXX` escape: the same string to a JSON reader, other bytes.
    fn escaped(s: &str) -> String {
        s.chars().map(|c| format!("\\u{:04x}", c as u32)).collect()
    }

    #[test]
    fn identifiers_are_used_as_the_provider_sent_them() {
        let with_key = |key: &str| Values::from(BTreeMap::from([(API_KEY.to_string(), key.to_string())]));
        // Short secrets (below the length that is scrubbed at all), and the real ones, inside the identifiers.
        for (key, token, pod, gpu) in [
            ("abc".to_string(), "abc".to_string(), "abc123xyz".to_string(), "NVIDIA abc".to_string()),
            (KEY.to_string(), TOKEN.to_string(), format!("p{KEY}"), format!("NVIDIA {KEY}")),
        ] {
            let values = with_key(&key);
            // A launch answered with a pod id that holds the worker's token succeeds, with that id.
            let f = Fixture::serve(vec![Answer::json(
                201,
                format!(r#"{{"id":"{pod}","status":"PROVISIONING"}}"#),
            )]);
            let launching = LaunchSpec { token: token.clone(), ..spec("NVIDIA L4@secure") };
            assert_eq!(runpod(&f).launch(&values, &launching).unwrap(), pod);
            let f = Fixture::serve(vec![Answer::json(
                201,
                format!(r#"{{"id":"p{token}","status":"PROVISIONING"}}"#),
            )]);
            assert_eq!(runpod(&f).launch(&values, &launching).unwrap(), format!("p{token}"));
            // `find` answers the id as it is, and terminating it sends DELETE for that id.
            let page = format!(
                r#"{{"pods":[{{"id":"{pod}","name":"folo-x","status":"RUNNING"}}],"pagination":{{}}}}"#
            );
            let f = Fixture::serve(vec![Answer::json(200, page), Answer::empty(204)]);
            let p = runpod(&f);
            let ids = p.find(&values, "folo-x").unwrap();
            assert_eq!(ids, std::slice::from_ref(&pod));
            p.terminate(&values, &ids[0]).unwrap();
            let r = &f.requests()[1];
            assert_eq!((r.method.as_str(), r.path.clone()), ("DELETE", format!("/v2/pods/{pod}")));
            // An offer's id goes back to the provider in a launch as the catalog named it.
            let gpus = format!(
                r#"{{"gpus":[{{"id":"{gpu}","name":"G","manufacturer":"NVIDIA","memory":24,"secure":true,
                    "community":false,"price":{{"secure":0.5}},"availability":"HIGH"}}]}}"#
            );
            let created = Answer::json(201, r#"{"id":"abc999","status":"PROVISIONING"}"#);
            let f =
                Fixture::serve(vec![Answer::json(200, gpus), Answer::json(200, r#"{"gpus":[]}"#), created]);
            let p = runpod(&f);
            let offer = p.offers(&values).unwrap().remove(0);
            assert_eq!(offer.id, format!("{gpu}@secure"));
            p.launch(&values, &LaunchSpec { token: token.clone(), ..spec(&offer.id) }).unwrap();
            assert_eq!(f.requests()[2].json()["gpu"]["id"], gpu.as_str());
        }
    }

    #[test]
    fn fields_of_successful_answers_never_carry_the_key_into_a_message_or_an_offer() {
        let key = escaped(KEY);
        let shows = |text: &str| text.contains(KEY);
        // A pod in a status FOLO does not know, spelled with the key.
        let f = Fixture::serve(vec![Answer::json(200, format!(r#"{{"id":"abc123xyz","status":"{key}"}}"#))]);
        assert_error(
            &runpod(&f).state(&values(), "abc123xyz").unwrap_err(),
            ErrorKind::Unavailable,
            "a status",
        );
        // A GPU type named with it: its name is shown (and logged, "launching <gpu>"); its id is the provider's
        // reference and is kept as it came (identifiers_are_used_as_the_provider_sent_them).
        let gpus = format!(
            r#"{{"gpus":[{{"id":"NVIDIA {key}","name":"{key}","manufacturer":"NVIDIA","memory":24,"secure":true,
                "community":false,"price":{{"secure":0.5}},"availability":"HIGH"}}]}}"#
        );
        let f = Fixture::serve(vec![Answer::json(200, gpus), Answer::json(200, r#"{"gpus":[]}"#)]);
        let offers = runpod(&f).offers(&values()).unwrap();
        assert!(offers.len() == 1 && !shows(&offers[0].gpu), "{offers:?}");
        // The account's email.
        let me = format!(r#"{{"data":{{"myself":{{"email":"{key}","clientBalance":1}}}}}}"#);
        let f = Fixture::serve(vec![Answer::json(200, NO_PODS), Answer::json(200, me)]);
        let label = runpod(&f).check(&values()).unwrap().label.unwrap();
        assert!(!shows(&label), "{label}");
    }

    #[test]
    fn secrets_echoed_in_any_spelling_are_not_repeated() {
        for (key, token) in [(KEY.to_string(), TOKEN.to_string()), (escaped(KEY), escaped(TOKEN))] {
            let refused = format!(
                r#"{{"title":"Bad Request","status":400,"detail":"bad key {key} and token {token}"}}"#
            );
            let invalid = format!(
                r#"{{"title":"Unprocessable Entity","status":422,"detail":"validation failed for {key}","errors":[
                    {{"message":"unexpected {token}","location":"body.env","value":{{"FOLO_TOKEN":"{token}"}}}}]}}"#
            );
            let f = Fixture::serve(vec![Answer::json(400, refused), Answer::json(422, invalid)]);
            let p = runpod(&f);
            let e = p.launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
            assert_error(&e, ErrorKind::Refused, "bad key [redacted] and token [redacted]");
            let e = p.launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
            assert_error(&e, ErrorKind::Unavailable, "validation failed for [redacted]");
            assert!(e.message.contains(r#""FOLO_TOKEN":"[redacted]""#), "{}", e.message);
        }
        // A long entry is shortened after the scrub: a token across the cut leaves no part of itself behind.
        let long = format!(r#"{{"status":422,"detail":"x","errors":["{}{TOKEN}"]}}"#, "a".repeat(290));
        let f = Fixture::serve(vec![Answer::json(422, long)]);
        let e = runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap_err();
        assert!(!e.message.contains(&TOKEN[..8]), "{}", e.message);
        // An answer that is not an error is read as it came, also where the token's text happens to be.
        let created = format!(r#"{{"id":"abc123xyz","name":"{TOKEN}","status":"PROVISIONING"}}"#);
        let f = Fixture::serve(vec![Answer::json(201, created)]);
        assert_eq!(runpod(&f).launch(&values(), &spec("NVIDIA L4@secure")).unwrap(), "abc123xyz");
    }
}
