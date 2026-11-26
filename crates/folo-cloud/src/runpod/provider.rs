//! [`RunPod`] as a [`Provider`]: check a connection, list offers, launch a pod, look it up, find it by name and
//! terminate it.

use std::collections::{BTreeMap, HashSet};

use folo_api::{CloudField, CloudOffer, CloudProvider};

use super::offers::{Cloud, parse_offer};
use super::{
    API_KEY, CreatePod, GpuRequest, IMAGE, IMAGE_ENV, MAX_PAGES, Pod, Problem, RunPod, SELF_TERMINATE,
};
use crate::bootstrap;
use crate::{
    Account, Error, ErrorKind, InstanceState, LaunchSpec, Provider, Result, Values, WORKER_BASE_URL_FIELD,
    WORKER_PORT,
};

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
        // Validate the resolved worker download base URL (whether from the connection field or
        // `FOLO_CLOUD_WORKER_BASE_URL`) before probing so embedded credentials or malformed URLs are never
        // requested or echoed into the persisted account note.
        let base_url = crate::worker_base_url(values, env!("CARGO_PKG_VERSION"));
        bootstrap::check_base_url(&base_url)?;
        // Listing pods needs what FOLO needs: a key that works and may see pods.
        let reply = self.api.get(&format!("{}/pods?limit=1", self.rest), key)?;
        if !reply.is_success() {
            return Err(self.failure(&reply));
        }
        // Best effort: a working key is a working connection, also when GraphQL does not answer.
        let mut account = self.account(key).unwrap_or_default();
        // A connection whose instances cannot install FOLO works, and every run on it would fail: say so here.
        account.note = self.download_problem(&base_url).map(|p| sentence(&p));
        Ok(account)
    }

    fn offers(&self, values: &Values) -> Result<Vec<CloudOffer>> {
        let key = api_key(values)?;
        // Fetch Secure and Community GPU catalogs concurrently; fall back to inline execution if thread creation
        // fails, and preserve Secure-first error precedence and deterministic sorting.
        let (secure, community) = std::thread::scope(|s| {
            let community = std::thread::Builder::new()
                .name("folo-runpod-catalog".into())
                .spawn_scoped(s, || self.catalog(key, Cloud::Community))
                .ok();
            let secure = self.catalog(key, Cloud::Secure);
            let community = match community {
                Some(handle) => match handle.join() {
                    Ok(res) => res,
                    Err(payload) => std::panic::resume_unwind(payload),
                },
                None => self.catalog(key, Cloud::Community),
            };
            (secure, community)
        });
        let mut offers = secure?;
        offers.extend(community?);
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
        // The worker's token is in the request too, and both transport errors and refusals must scrub it.
        let reply = self.api.post_with_secrets(&format!("{}/pods", self.rest), key, &body, &[&spec.token])?;
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
        let mut seen_ids = HashSet::new();
        let mut seen_cursors = HashSet::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_PAGES {
            let page = self.pods(key, cursor.as_deref())?;
            for p in page.pods {
                if p.name == name && p.status != "TERMINATED" {
                    if check_pod_id(&p.id).is_err() {
                        return Err(Error::new(
                            ErrorKind::Unavailable,
                            "RunPod listed a live pod with an invalid id",
                        ));
                    }
                    if seen_ids.insert(p.id.clone()) {
                        ids.push(p.id);
                    }
                }
            }
            match page.pagination.next_cursor.filter(|c| !c.is_empty()) {
                Some(next) if seen_cursors.insert(next.clone()) => cursor = Some(next),
                Some(_) => {
                    return Err(Error::new(
                        ErrorKind::Unavailable,
                        "RunPod returned a repeated pagination cursor while listing pods",
                    ));
                }
                None => return Ok(ids),
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
