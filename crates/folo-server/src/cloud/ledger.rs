//! The instance ledger: `cloud/instances/<id>.json`, one [`CloudInstance`] per instance FOLO launched (docs/CLOUD.md,
//! *Instances are never left behind*). A record is written (`launching`, with the instance's unique name) before the
//! provider is asked for the instance, and rewritten atomically under the cloud lock at every transition, so a
//! process that dies at any point leaves a record the next one can act on, and two processes on one data dir (a
//! server and `folo cloud terminate`) never lose each other's changes. Records of instances that ended more than
//! [`KEEP_ENDED`] ago are removed.
//!
//! Next to the instance a record keeps when its worker first answered (`booted`), which the API does not show: from
//! then, an instance that terminates itself has a known latest end ([`Ledger::last`]).

use std::path::{Path, PathBuf};
use std::time::Duration;

use folo_api::{CloudInstance, CloudInstanceStatus, CloudOffer};
use serde::{Deserialize, Serialize};

use super::files;
use crate::error::{Error, Result};
use crate::util::{fnv1a, parse_rfc3339, rfc3339, unique_id, unix_now};

/// How long the records of instances that ended are kept.
pub const KEEP_ENDED: Duration = Duration::from_secs(7 * 24 * 3600);

/// The ledger of one data dir.
pub(crate) struct Ledger {
    dir: PathBuf,
    lock: PathBuf,
    /// The providers whose instances terminate themselves after their deadline
    /// (`CloudProvider::self_terminates`), by id: what bounds an instance's life when nobody saw its end.
    self_terminating: Vec<String>,
}

/// A record as it is stored: the instance as the API shows it, and what only the ledger needs.
#[derive(Serialize, Deserialize)]
struct Record {
    #[serde(flatten)]
    instance: CloudInstance,
    /// RFC 3339 time its worker first answered; absent until then (and in records of before this field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    booted: Option<String>,
}

/// `404 instance_not_found`.
pub(crate) fn not_found(id: &str) -> Error {
    Error::not_found("instance", id)
        .hint("GET /api/cloud/instances (`folo cloud instances`, MCP cloud_instances) lists them")
}

/// What `instance` has cost so far (or in all, once it ended): its price per hour times the time from the launch
/// request to the confirmed end, or to `now` and at most to `last` (the latest it can have lived, when that is
/// known); `None` without a price.
pub(crate) fn cost(instance: &CloudInstance, now: f64, last: Option<f64>) -> Option<f64> {
    let price = instance.offer.price_per_hour?;
    let start = parse_rfc3339(&instance.created)?;
    let end = instance
        .ended
        .as_deref()
        .and_then(parse_rfc3339)
        .unwrap_or_else(|| last.map_or(now, |last| now.min(last)));
    Some(((end - start).max(0.0) / 3600.0 * price * 1e4).round() / 1e4)
}

impl Ledger {
    /// The ledger in `cloud_dir`, guarded by the cloud lock `lock`; `self_terminating` are the ids of the providers
    /// whose instances terminate themselves after their deadline.
    pub fn new(cloud_dir: &Path, lock: PathBuf, self_terminating: Vec<String>) -> Ledger {
        Ledger { dir: cloud_dir.join("instances"), lock, self_terminating }
    }

    /// The latest the instance of `r` can have lived, when that is known: an instance of a provider that arms it
    /// to terminate itself is gone its time limit and [`folo_cloud::bootstrap::GRACE_S`] after its setup started,
    /// whoever watched, and its setup had started when its worker answered (`booted`). So an instance that ended
    /// while nothing followed it (this machine was off, the server crashed) is not counted as billed until
    /// someone looked again. `None` for an instance whose worker never answered (when its setup started, if it
    /// did, is not known) and for a provider that does not arm its instances: those count until their end is
    /// confirmed.
    fn last(&self, r: &Record) -> Option<f64> {
        let i = &r.instance;
        if !self.self_terminating.contains(&i.provider) {
            return None;
        }
        let booted = r.booted.as_deref().and_then(parse_rfc3339)?;
        // The time limit: the deadline is set, counted from then, just before the record is created.
        let limit = i.deadline.as_deref().and_then(parse_rfc3339)? - parse_rfc3339(&i.created)?;
        Some(booted + limit.max(0.0) + folo_cloud::bootstrap::GRACE_S as f64)
    }

    /// The instance of `r` as the API shows it at `now`: with its cost so far.
    fn shown(&self, r: Record, now: f64) -> CloudInstance {
        let last = self.last(&r);
        let mut i = r.instance;
        i.cost_usd = cost(&i, now, last);
        i
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn lock(&self) -> Result<std::fs::File> {
        files::lock(&self.lock).map_err(|e| Error::io(format!("locking {}: {e}", self.lock.display())))
    }

    fn read(&self, id: &str) -> Result<Record> {
        let path = self.path(id);
        let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => not_found(id),
            _ => Error::io(format!("reading {}: {e}", path.display())),
        })?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::internal(format!("{} is malformed: {e}", path.display())))
    }

    fn write(&self, r: &Record) -> Result<()> {
        let path = self.path(&r.instance.id);
        let bytes = serde_json::to_vec_pretty(r).expect("instances serialize");
        files::write_private(&path, &bytes).map_err(|e| Error::io(format!("writing {}: {e}", path.display())))
    }

    /// A new `launching` record for an instance of `offer` for run `run_id` on `connection_id`, named
    /// `folo-<run id>-<nonce>`, with the run's `deadline`.
    pub fn create(
        &self,
        connection_id: &str,
        provider: &str,
        run_id: &str,
        offer: &CloudOffer,
        deadline: Option<String>,
    ) -> Result<CloudInstance> {
        let nonce = files::random_hex(2).map_err(|e| Error::internal(e.to_string()))?;
        let _lock = self.lock()?;
        std::fs::create_dir_all(&self.dir)
            .map_err(|e| Error::io(format!("creating {}: {e}", self.dir.display())))?;
        let seed = fnv1a(run_id.as_bytes()) ^ unix_now().to_bits();
        let id = unique_id("ci", seed, |id| self.path(id).exists());
        let instance = CloudInstance {
            id,
            connection_id: connection_id.into(),
            provider: provider.into(),
            provider_ref: None,
            name: format!("folo-{run_id}-{nonce}"),
            run_id: Some(run_id.into()),
            offer: offer.clone(),
            status: CloudInstanceStatus::Launching,
            created: rfc3339(unix_now()),
            ended: None,
            deadline,
            cost_usd: None,
            error: None,
        };
        let record = Record { instance, booted: None };
        self.write(&record)?;
        Ok(record.instance)
    }

    /// Record `id`, with its cost so far.
    pub fn get(&self, id: &str) -> Result<CloudInstance> {
        Ok(self.shown(self.read(id)?, unix_now()))
    }

    /// Changes record `id` under the cloud lock (re-read first, so another process's change is not lost) and
    /// writes it; its cost is brought up to date, and an instance that `terminated` or was `lost` gets its end
    /// (now, or the latest it can have lived when that is known and was earlier).
    pub fn update(&self, id: &str, change: impl FnOnce(&mut CloudInstance)) -> Result<CloudInstance> {
        self.change(id, |r| change(&mut r.instance))
    }

    /// The worker of instance `id` answers: its record says when (the first time only), which bounds how long the
    /// instance can live ([`Ledger::last`]).
    pub fn booted(&self, id: &str) -> Result<CloudInstance> {
        self.booted_at(id, unix_now())
    }

    fn booted_at(&self, id: &str, at: f64) -> Result<CloudInstance> {
        self.change(id, |r| {
            r.booted.get_or_insert_with(|| rfc3339(at));
        })
    }

    /// [`Ledger::update`] on the whole record.
    fn change(&self, id: &str, change: impl FnOnce(&mut Record)) -> Result<CloudInstance> {
        let _lock = self.lock()?;
        let mut r = self.read(id)?;
        change(&mut r);
        let now = unix_now();
        if !r.instance.status.is_alive() && r.instance.ended.is_none() {
            let end = self.last(&r).map_or(now, |last| now.min(last));
            r.instance.ended = Some(rfc3339(end));
        }
        r.instance.cost_usd = cost(&r.instance, now, self.last(&r));
        self.write(&r)?;
        Ok(r.instance)
    }

    /// Removes record `id` (a launch that created nothing).
    pub fn remove(&self, id: &str) {
        let _lock = self.lock();
        if let Err(e) = std::fs::remove_file(self.path(id))
            && e.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!("removing the ledger record {id}: {e}");
        }
    }

    /// Every record with its cost so far: the instances that are alive, then those that ended, newest first.
    /// Records that ended more than [`KEEP_ENDED`] ago are removed on the way.
    pub fn list(&self) -> Vec<CloudInstance> {
        let Ok(entries) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let now = unix_now();
        let mut out = Vec::new();
        for path in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
            let Some(id) = path.file_name().and_then(|n| n.to_str()).and_then(|n| n.strip_suffix(".json"))
            else {
                continue;
            };
            if id.starts_with('.') {
                continue;
            }
            match self.read(id) {
                Ok(r)
                    if r.instance
                        .ended
                        .as_deref()
                        .and_then(parse_rfc3339)
                        .is_some_and(|t| now - t > KEEP_ENDED.as_secs_f64()) =>
                {
                    self.remove(id);
                }
                Ok(r) => out.push(self.shown(r, now)),
                Err(e) => tracing::warn!("skipping {}: {}", path.display(), e.body.message),
            }
        }
        out.sort_by(|a, b| {
            (b.status.is_alive(), &b.created, &b.id).cmp(&(a.status.is_alive(), &a.created, &a.id))
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_api::CloudAvailability;
    use folo_test_support::TestDir;

    fn offer(price: Option<f64>) -> CloudOffer {
        CloudOffer {
            id: "o".into(),
            gpu: "G".into(),
            gpu_count: 1,
            vram_bytes: 0,
            price_per_hour: price,
            region: None,
            detail: None,
            availability: CloudAvailability::High,
        }
    }

    #[test]
    fn records_follow_their_instance_and_old_ones_go() {
        let dir = TestDir::new("cloud-ledger");
        let ledger = Ledger::new(&dir.join("cloud"), dir.join("cloud").join(".lock"), Vec::new());
        let a = ledger.create("cc_1", "p", "run_1", &offer(Some(3.6)), None).unwrap();
        assert!(a.name.starts_with("folo-run_1-") && a.name.len() == "folo-run_1-".len() + 4, "{}", a.name);
        assert_eq!((a.status, a.provider_ref.as_deref()), (CloudInstanceStatus::Launching, None));
        let b = ledger.create("cc_1", "p", "run_2", &offer(None), None).unwrap();
        let ended = ledger
            .update(&b.id, |i| {
                i.provider_ref = Some("r".into());
                i.status = CloudInstanceStatus::Terminated;
            })
            .unwrap();
        assert!(ended.ended.is_some() && ended.cost_usd.is_none());
        assert_eq!(ledger.get(&b.id).unwrap().provider_ref.as_deref(), Some("r"));
        let listed: Vec<String> = ledger.list().into_iter().map(|i| i.id).collect();
        assert_eq!(listed, [a.id.clone(), b.id.clone()], "alive first");
        assert_eq!(ledger.get("ci_none").unwrap_err().body.code, "instance_not_found");

        // $3.60 an hour for an hour is $3.60.
        let mut hour = ledger.get(&a.id).unwrap();
        hour.created = rfc3339(1000.0);
        hour.ended = Some(rfc3339(4600.0));
        assert_eq!(cost(&hour, 0.0, None), Some(3.6));

        ledger
            .update(&b.id, |i| i.ended = Some(rfc3339(unix_now() - KEEP_ENDED.as_secs_f64() - 60.0)))
            .unwrap();
        assert_eq!(ledger.list().len(), 1);
        assert_eq!(ledger.get(&b.id).unwrap_err().body.code, "instance_not_found", "pruned");
    }

    #[test]
    fn an_instance_that_terminates_itself_is_not_billed_past_the_latest_it_can_have_lived() {
        let dir = TestDir::new("cloud-ledger-last");
        let cloud = dir.join("cloud");
        let grace = folo_cloud::bootstrap::GRACE_S as f64;
        // Launched ten hours ago with an hour's time limit, and nobody has looked since.
        let now = unix_now().floor();
        let (created, deadline) = (now - 36_000.0, now - 32_400.0);
        let forgotten = |ledger: &Ledger, provider: &str, status: CloudInstanceStatus| {
            let i =
                ledger.create("cc_1", provider, "run_1", &offer(Some(3.6)), Some(rfc3339(deadline))).unwrap();
            ledger
                .update(&i.id, |i| {
                    i.created = rfc3339(created);
                    i.status = status;
                })
                .unwrap()
        };
        let usd = |seconds: f64| Some((seconds / 3600.0 * 3.6 * 1e4).round() / 1e4);
        let since = |i: &CloudInstance| now - 36_000.0 <= unix_now() && i.cost_usd.unwrap() >= 35.99;

        // A provider that arms its instances, and a worker that answered four minutes after the launch request:
        // the instance is gone an hour and the grace after that at the latest, whatever `now` is.
        let ledger = Ledger::new(&cloud, cloud.join(".lock"), vec!["arming".into()]);
        let alive = forgotten(&ledger, "arming", CloudInstanceStatus::Running);
        assert!(since(&alive), "not bounded before its worker answered: {:?}", alive.cost_usd);
        let alive = ledger.booted_at(&alive.id, created + 240.0).unwrap();
        assert_eq!(alive.cost_usd, usd(240.0 + 3_600.0 + grace));
        assert_eq!(ledger.list()[0].cost_usd, usd(240.0 + 3_600.0 + grace));
        // The first answer counts: a later one (a restart that follows the run again) does not move the bound.
        assert_eq!(ledger.booted(&alive.id).unwrap().cost_usd, usd(240.0 + 3_600.0 + grace));
        // Found gone, through `terminating`: it ended when it terminated itself at the latest, not when someone
        // looked.
        ledger.update(&alive.id, |i| i.status = CloudInstanceStatus::Terminating).unwrap();
        let ended = ledger.update(&alive.id, |i| i.status = CloudInstanceStatus::Terminated).unwrap();
        assert_eq!(ended.ended.as_deref().and_then(parse_rfc3339), Some(created + 240.0 + 3_600.0 + grace));
        assert_eq!(ended.cost_usd, usd(240.0 + 3_600.0 + grace));
        // The file keeps the time next to the instance, and the API's instance does not carry it.
        let file =
            std::fs::read_to_string(cloud.join("instances").join(format!("{}.json", alive.id))).unwrap();
        assert!(file.contains(r#""booted""#) && file.contains(r#""status": "terminated""#), "{file}");
        assert!(!serde_json::to_string(&ended).unwrap().contains("booted"));

        // An instance that ends before then ends now.
        let early =
            ledger.create("cc_1", "arming", "run_2", &offer(None), Some(rfc3339(now + 3_600.0))).unwrap();
        ledger.booted(&early.id).unwrap();
        let early = ledger.update(&early.id, |i| i.status = CloudInstanceStatus::Terminated).unwrap();
        assert!((early.ended.as_deref().and_then(parse_rfc3339).unwrap() - now).abs() < 5.0);

        // An instance whose worker never answered: when its setup started, if it did, nobody knows. It counts
        // until its end is confirmed, as every instance of a provider that does not arm them does.
        let silent = forgotten(&ledger, "arming", CloudInstanceStatus::Starting);
        assert!(since(&silent), "{:?}", silent.cost_usd);
        let other = TestDir::new("cloud-ledger-unarmed");
        let unarmed = Ledger::new(&other.join("cloud"), other.join("cloud").join(".lock"), Vec::new());
        let alive = forgotten(&unarmed, "arming", CloudInstanceStatus::Running);
        let alive = unarmed.booted_at(&alive.id, created + 240.0).unwrap();
        assert!(since(&alive), "{:?}", alive.cost_usd);
    }

    #[test]
    fn records_of_before_the_booted_time_still_read() {
        let dir = TestDir::new("cloud-ledger-old");
        let cloud = dir.join("cloud");
        let ledger = Ledger::new(&cloud, cloud.join(".lock"), vec!["p".into()]);
        let i = ledger.create("cc_1", "p", "run_1", &offer(Some(1.0)), None).unwrap();
        // As a server of before wrote it: the instance alone.
        let path = cloud.join("instances").join(format!("{}.json", i.id));
        std::fs::write(&path, serde_json::to_vec_pretty(&i).unwrap()).unwrap();
        assert_eq!(ledger.get(&i.id).unwrap().id, i.id);
        assert_eq!(ledger.list().len(), 1);
    }
}
