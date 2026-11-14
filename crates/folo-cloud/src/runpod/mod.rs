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
//!
//! This file holds [`RunPod`], its API calls and answer types; `provider.rs` implements
//! [`Provider`](crate::Provider) on it, `offers.rs` turns catalog GPU types into offers, and `tests.rs` runs it
//! against a scripted server.

mod offers;
mod provider;

use std::collections::BTreeMap;

use folo_api::CloudOffer;
use serde::{Deserialize, Deserializer, Serialize};

use crate::http::{Api, Reply};
use crate::{Account, Error, ErrorKind, Result};
use offers::{Cloud, offer};

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
mod tests;
