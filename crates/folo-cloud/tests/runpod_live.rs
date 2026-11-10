//! The RunPod provider against the real service (docs/CLOUD.md, Limits). Ignored by default; run with
//!
//! ```text
//! RUNPOD_API_KEY=rpa_... cargo test -p folo-cloud --test runpod_live -- --ignored --nocapture
//! ```
//!
//! `check` and `offers` only read the account and the catalog: they cost nothing. With `FOLO_LIVE_LAUNCH=1` as well,
//! `launch_wait_and_terminate` rents the cheapest offer that has stock for a few minutes (typically a few US
//! cents; the pod is terminated as soon as it runs, also when the test fails, and terminates itself after 25
//! minutes in any case). The pod's worker installs the release of this crate's version, which need not exist:
//! the test only waits for RunPod to report the pod running, so it launches without asking for the download
//! first (`RunPod::with_urls`; `RunPod::new` refuses to rent a pod that could not install FOLO).
//!
//! The whole lifecycle (a worker that boots, finds its GPU, trains and is released) is checked by a real run:
//! docs/CLOUD.md, *Checking a provider against the real service*.
//!
//! Without `RUNPOD_API_KEY` each test says so and passes.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use folo_api::CloudAvailability;
use folo_cloud::runpod::{self, RunPod};
use folo_cloud::{InstanceState, LaunchSpec, Provider, Values};

/// The connection of `RUNPOD_API_KEY`, if it is set.
fn connection() -> Option<Values> {
    match std::env::var("RUNPOD_API_KEY") {
        Ok(key) if !key.trim().is_empty() => {
            Some(Values::from(BTreeMap::from([("api_key".to_string(), key)])))
        }
        _ => {
            eprintln!("skipped: set RUNPOD_API_KEY to run the RunPod live tests");
            None
        }
    }
}

#[test]
#[ignore = "needs RUNPOD_API_KEY and the network"]
fn check() {
    let Some(values) = connection() else { return };
    let account = RunPod::new().check(&values).unwrap();
    eprintln!("account: {:?}, balance: {:?}", account.label, account.balance_usd);
}

#[test]
#[ignore = "needs RUNPOD_API_KEY and the network"]
fn offers() {
    let Some(values) = connection() else { return };
    let offers = RunPod::new().offers(&values).unwrap();
    assert!(!offers.is_empty(), "RunPod offers no NVIDIA GPU");
    for o in &offers {
        eprintln!(
            "{:<45} {:>6.2} $/h {:>4} GiB {:?}",
            o.id,
            o.price_per_hour.unwrap(),
            o.vram_bytes >> 30,
            o.availability
        );
        assert!(o.price_per_hour.is_some_and(|p| p > 0.0) && o.vram_bytes > 0, "{o:?}");
    }
    let prices: Vec<f64> = offers.iter().filter_map(|o| o.price_per_hour).collect();
    assert!(prices.windows(2).all(|w| w[0] <= w[1]), "not cheapest first");
}

/// Terminates its pod when dropped: at the end of the test, also when it panics.
struct Terminate<'a> {
    provider: &'a RunPod,
    values: &'a Values,
    pod: String,
}

impl Drop for Terminate<'_> {
    fn drop(&mut self) {
        for attempt in 1..=5 {
            match self.provider.terminate(self.values, &self.pod) {
                Ok(()) => {
                    eprintln!("terminated pod {}", self.pod);
                    return;
                }
                Err(e) => eprintln!("terminating pod {} (attempt {attempt}): {e}", self.pod),
            }
            std::thread::sleep(Duration::from_secs(5));
        }
        eprintln!("COULD NOT TERMINATE pod {}: terminate it in the RunPod console", self.pod);
    }
}

#[test]
#[ignore = "needs RUNPOD_API_KEY, FOLO_LIVE_LAUNCH=1 and the network; rents a GPU for a few minutes"]
fn launch_wait_and_terminate() {
    let Some(values) = connection() else { return };
    if std::env::var("FOLO_LIVE_LAUNCH").as_deref() != Ok("1") {
        eprintln!("skipped: set FOLO_LIVE_LAUNCH=1 to rent a pod");
        return;
    }
    let provider = RunPod::with_urls(runpod::API_URL, runpod::GRAPHQL_URL, runpod::PROXY_URL);
    let offers = provider.offers(&values).unwrap();
    let offer = offers
        .iter()
        .find(|o| matches!(o.availability, CloudAvailability::High | CloudAvailability::Medium))
        .or_else(|| offers.iter().find(|o| o.availability != CloudAvailability::None))
        .expect("no offer has stock right now");
    let name = format!("folo-live-test-{}", std::process::id());
    let spec = LaunchSpec {
        name: name.clone(),
        offer_id: offer.id.clone(),
        disk_gb: 20,
        token: format!("live-{}", std::process::id()),
        lifetime_s: 900,
        worker_base_url: folo_cloud::worker_base_url(&values, env!("CARGO_PKG_VERSION")),
    };
    eprintln!("launching {} ({:?} $/h) as {name}", offer.id, offer.price_per_hour);
    let pod = provider.launch(&values, &spec).unwrap();
    let _guard = Terminate { provider: &provider, values: &values, pod: pod.clone() };
    assert_eq!(provider.find(&values, &name).unwrap(), std::slice::from_ref(&pod));

    let start = Instant::now();
    let url = loop {
        match provider.state(&values, &pod).unwrap() {
            InstanceState::Running { url } => break url,
            InstanceState::Starting => {}
            other => panic!("pod {pod} is {other:?} before it ran"),
        }
        assert!(start.elapsed() < Duration::from_secs(600), "pod {pod} did not run within 10 minutes");
        std::thread::sleep(Duration::from_secs(5));
    };
    eprintln!("pod {pod} runs after {:?}: {url}", start.elapsed());
    assert!(url.starts_with(&format!("https://{pod}-7878.proxy.runpod.net")), "{url}");
}
