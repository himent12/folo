//! The instance ledger: `cloud/instances/<id>.json`, one [`CloudInstance`] per instance FOLO launched (docs/CLOUD.md,
//! *Instances are never left behind*). A record is written (`launching`, with the instance's unique name) before the
//! provider is asked for the instance, and rewritten atomically under the cloud lock at every transition, so a
//! process that dies at any point leaves a record the next one can act on, and two processes on one data dir (a
//! server and `folo cloud terminate`) never lose each other's changes. Records of instances that ended more than
//! [`KEEP_ENDED`] ago are removed.
//!
//! Next to the instance a record keeps its time limit and when its worker first answered, which the API does not
//! show: with them, the cost of an instance that terminated itself while nobody watched is estimated to when it
//! should have, not to when someone looked again ([`Ledger::last`]).

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
    /// The time limit the instance was launched with, in seconds: what its own watchdog counts from the start of
    /// its setup. Absent when it has none (and in records of before this field).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    lifetime_s: Option<f64>,
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

    /// When the instance of `r` should have terminated itself at the latest, where that can be told: an instance
    /// of a provider that arms it does so its time limit and [`folo_cloud::bootstrap::GRACE_S`] after its setup
    /// started (and keeps trying for [`folo_cloud::bootstrap::TERMINATE_TRIES_S`]), whoever watches, and its
    /// setup had started when its worker answered (`booted`). `None` for an instance whose worker never answered
    /// (when its setup started, if it did, is not known), one without a time limit on record, and a provider
    /// that does not arm its instances.
    ///
    /// It is what the cost of an instance that nobody saw end is estimated to (this machine was off, the server
    /// crashed): better than counting until someone looked again, and not a proof. An instance that could not
    /// reach its provider's API to terminate itself is still there, which is why its end is only put here when
    /// the provider says it is gone ([`Ledger::terminated`]).
    fn last(&self, r: &Record) -> Option<f64> {
        if !self.self_terminating.contains(&r.instance.provider) {
            return None;
        }
        let booted = r.booted.as_deref().and_then(parse_rfc3339)?;
        let own = folo_cloud::bootstrap::GRACE_S + folo_cloud::bootstrap::TERMINATE_TRIES_S;
        Some(booted + r.lifetime_s?.max(0.0) + own as f64)
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
    /// `folo-<run id>-<nonce>`, with the run's `deadline` and the time limit (`lifetime`) the instance is launched
    /// with.
    pub fn create(
        &self,
        connection_id: &str,
        provider: &str,
        run_id: &str,
        offer: &CloudOffer,
        deadline: Option<String>,
        lifetime: Option<Duration>,
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
        let record = Record { instance, booted: None, lifetime_s: lifetime.map(|l| l.as_secs_f64()) };
        self.write(&record)?;
        Ok(record.instance)
    }

    /// Record `id`, with its cost so far.
    pub fn get(&self, id: &str) -> Result<CloudInstance> {
        Ok(self.shown(self.read(id)?, unix_now()))
    }

    /// Changes record `id` under the cloud lock (re-read first, so another process's change is not lost) and
    /// writes it; its cost is brought up to date, and an instance that `terminated` or was `lost` gets its end:
    /// now.
    pub fn update(&self, id: &str, change: impl FnOnce(&mut CloudInstance)) -> Result<CloudInstance> {
        self.change(id, |r| change(&mut r.instance))
    }

    /// The provider confirmed that instance `id` is gone: `terminated`, without an error. `was_gone` says the
    /// provider no longer had it when it was asked, so nobody saw it end: its end is then put at when it should
    /// have terminated itself ([`Ledger::last`]) when that is known and was earlier. An instance that was still
    /// there ended now.
    pub fn terminated(&self, id: &str, was_gone: bool) -> Result<CloudInstance> {
        let now = unix_now();
        self.change(id, |r| {
            r.instance.status = CloudInstanceStatus::Terminated;
            r.instance.error = None;
            if r.instance.ended.is_none() {
                let end = self.last(r).filter(|_| was_gone).map_or(now, |last| now.min(last));
                r.instance.ended = Some(rfc3339(end));
            }
        })
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
            r.instance.ended = Some(rfc3339(now));
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
        let a = ledger.create("cc_1", "p", "run_1", &offer(Some(3.6)), None, None).unwrap();
        assert!(a.name.starts_with("folo-run_1-") && a.name.len() == "folo-run_1-".len() + 4, "{}", a.name);
        assert_eq!((a.status, a.provider_ref.as_deref()), (CloudInstanceStatus::Launching, None));
        let b = ledger.create("cc_1", "p", "run_2", &offer(None), None, None).unwrap();
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
    fn an_instance_nobody_saw_end_is_counted_to_when_it_should_have_terminated_itself() {
        let dir = TestDir::new("cloud-ledger-last");
        let cloud = dir.join("cloud");
        let own = (folo_cloud::bootstrap::GRACE_S + folo_cloud::bootstrap::TERMINATE_TRIES_S) as f64;
        // Launched ten hours ago with an hour's time limit, and nobody has looked since.
        let now = unix_now().floor();
        let created = now - 36_000.0;
        let hour = Some(Duration::from_secs(3_600));
        let forgotten = |ledger: &Ledger, status: CloudInstanceStatus, lifetime: Option<Duration>| {
            let deadline = Some(rfc3339(created + 3_600.0));
            let i = ledger.create("cc_1", "arming", "run_1", &offer(Some(3.6)), deadline, lifetime).unwrap();
            ledger
                .update(&i.id, |i| {
                    i.created = rfc3339(created);
                    i.status = status;
                })
                .unwrap()
        };
        let usd = |seconds: f64| Some((seconds / 3600.0 * 3.6 * 1e4).round() / 1e4);
        let ten_hours = |i: &CloudInstance| i.cost_usd.unwrap() >= 35.99;
        let end = |i: &CloudInstance| i.ended.as_deref().and_then(parse_rfc3339).unwrap();

        // A provider that arms its instances, and a worker that answered four minutes after the launch request:
        // the instance should be gone an hour and its own grace after that, whatever `now` is.
        let ledger = Ledger::new(&cloud, cloud.join(".lock"), vec!["arming".into()]);
        let alive = forgotten(&ledger, CloudInstanceStatus::Running, hour);
        assert!(ten_hours(&alive), "no estimate before its worker answered: {:?}", alive.cost_usd);
        let alive = ledger.booted_at(&alive.id, created + 240.0).unwrap();
        let should = 240.0 + 3_600.0 + own;
        assert_eq!(alive.cost_usd, usd(should));
        assert_eq!(ledger.list()[0].cost_usd, usd(should));
        // The first answer counts: a later one (a restart that follows the run again) does not move it.
        assert_eq!(ledger.booted(&alive.id).unwrap().cost_usd, usd(should));
        // The provider no longer had it when it was asked: it ended when it should have, not when someone looked.
        ledger.update(&alive.id, |i| i.status = CloudInstanceStatus::Terminating).unwrap();
        let gone = ledger.terminated(&alive.id, true).unwrap();
        assert_eq!(
            (gone.status, end(&gone), gone.cost_usd),
            (CloudInstanceStatus::Terminated, created + should, usd(should))
        );
        // The file keeps the two times next to the instance, and the API's instance does not carry them.
        let file =
            std::fs::read_to_string(cloud.join("instances").join(format!("{}.json", alive.id))).unwrap();
        assert!(file.contains(r#""booted""#) && file.contains(r#""lifetime_s": 3600.0"#), "{file}");
        let shown = serde_json::to_string(&gone).unwrap();
        assert!(!shown.contains("booted") && !shown.contains("lifetime_s"), "{shown}");

        // The provider still had it (its own termination failed, or it was not time yet): it ended now.
        let there = forgotten(&ledger, CloudInstanceStatus::Running, hour);
        ledger.booted_at(&there.id, created + 240.0).unwrap();
        let there = ledger.terminated(&there.id, false).unwrap();
        assert!((end(&there) - now).abs() < 5.0 && ten_hours(&there), "{there:?}");
        // So does one that is terminated before its time, and one that changes status any other way.
        let early = ledger.create("cc_1", "arming", "run_2", &offer(None), None, hour).unwrap();
        ledger.booted(&early.id).unwrap();
        assert!((end(&ledger.terminated(&early.id, true).unwrap()) - now).abs() < 5.0);
        let lost = forgotten(&ledger, CloudInstanceStatus::Running, hour);
        ledger.booted_at(&lost.id, created + 240.0).unwrap();
        let lost = ledger.update(&lost.id, |i| i.status = CloudInstanceStatus::Lost).unwrap();
        assert!((end(&lost) - now).abs() < 5.0);

        // No estimate without what it takes: a worker that never answered (when its setup started, if it did,
        // nobody knows), no time limit on record, a provider that does not arm its instances.
        let silent = forgotten(&ledger, CloudInstanceStatus::Starting, hour);
        assert!(ten_hours(&silent) && ten_hours(&ledger.terminated(&silent.id, true).unwrap()));
        let unlimited = forgotten(&ledger, CloudInstanceStatus::Running, None);
        assert!(ten_hours(&ledger.booted_at(&unlimited.id, created + 240.0).unwrap()));
        let other = TestDir::new("cloud-ledger-unarmed");
        let unarmed = Ledger::new(&other.join("cloud"), other.join("cloud").join(".lock"), Vec::new());
        let alive = forgotten(&unarmed, CloudInstanceStatus::Running, hour);
        assert!(ten_hours(&unarmed.booted_at(&alive.id, created + 240.0).unwrap()));
    }

    #[test]
    fn records_of_before_the_ledger_kept_more_than_the_instance_still_read() {
        let dir = TestDir::new("cloud-ledger-old");
        let cloud = dir.join("cloud");
        let ledger = Ledger::new(&cloud, cloud.join(".lock"), vec!["p".into()]);
        let i = ledger.create("cc_1", "p", "run_1", &offer(Some(1.0)), None, None).unwrap();
        // As a server of before wrote it: the instance alone.
        let path = cloud.join("instances").join(format!("{}.json", i.id));
        std::fs::write(&path, serde_json::to_vec_pretty(&i).unwrap()).unwrap();
        assert_eq!(ledger.get(&i.id).unwrap().id, i.id);
        assert_eq!(ledger.list().len(), 1);
        // And a record of now reads as the instance alone, as a server of before reads it.
        ledger.booted(&i.id).unwrap();
        let old: CloudInstance = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        assert_eq!(old.id, i.id);
    }
}
