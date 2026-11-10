//! The instance ledger: `cloud/instances/<id>.json`, one [`CloudInstance`] per instance FOLO launched (docs/CLOUD.md,
//! *Instances are never left behind*). A record is written (`launching`, with the instance's unique name) before the
//! provider is asked for the instance, and rewritten atomically under the cloud lock at every transition, so a
//! process that dies at any point leaves a record the next one can act on, and two processes on one data dir (a
//! server and `folo cloud terminate`) never lose each other's changes. Records of instances that ended more than
//! [`KEEP_ENDED`] ago are removed.

use std::path::{Path, PathBuf};
use std::time::Duration;

use folo_api::{CloudInstance, CloudInstanceStatus, CloudOffer};

use super::files;
use crate::error::{Error, Result};
use crate::util::{fnv1a, parse_rfc3339, rfc3339, unique_id, unix_now};

/// How long the records of instances that ended are kept.
pub const KEEP_ENDED: Duration = Duration::from_secs(7 * 24 * 3600);

/// The ledger of one data dir.
pub(crate) struct Ledger {
    dir: PathBuf,
    lock: PathBuf,
}

/// `404 instance_not_found`.
pub(crate) fn not_found(id: &str) -> Error {
    Error::not_found("instance", id)
        .hint("GET /api/cloud/instances (`folo cloud instances`, MCP cloud_instances) lists them")
}

/// What `instance` has cost so far (or in all, once it ended): its price per hour times the time from the launch
/// request to the confirmed end (or `now`); `None` without a price.
pub(crate) fn cost(instance: &CloudInstance, now: f64) -> Option<f64> {
    let price = instance.offer.price_per_hour?;
    let start = parse_rfc3339(&instance.created)?;
    let end = instance.ended.as_deref().and_then(parse_rfc3339).unwrap_or(now);
    Some(((end - start).max(0.0) / 3600.0 * price * 1e4).round() / 1e4)
}

impl Ledger {
    /// The ledger in `cloud_dir`, guarded by the cloud lock `lock`.
    pub fn new(cloud_dir: &Path, lock: PathBuf) -> Ledger {
        Ledger { dir: cloud_dir.join("instances"), lock }
    }

    fn path(&self, id: &str) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }

    fn lock(&self) -> Result<std::fs::File> {
        files::lock(&self.lock).map_err(|e| Error::io(format!("locking {}: {e}", self.lock.display())))
    }

    fn read(&self, id: &str) -> Result<CloudInstance> {
        let path = self.path(id);
        let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => not_found(id),
            _ => Error::io(format!("reading {}: {e}", path.display())),
        })?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::internal(format!("{} is malformed: {e}", path.display())))
    }

    fn write(&self, i: &CloudInstance) -> Result<()> {
        let path = self.path(&i.id);
        let bytes = serde_json::to_vec_pretty(i).expect("instances serialize");
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
        self.write(&instance)?;
        Ok(instance)
    }

    /// Record `id`, with its cost so far.
    pub fn get(&self, id: &str) -> Result<CloudInstance> {
        let mut i = self.read(id)?;
        i.cost_usd = cost(&i, unix_now());
        Ok(i)
    }

    /// Changes record `id` under the cloud lock (re-read first, so another process's change is not lost) and
    /// writes it; its cost is brought up to date, and an instance that `terminated` or was `lost` gets its end.
    pub fn update(&self, id: &str, change: impl FnOnce(&mut CloudInstance)) -> Result<CloudInstance> {
        let _lock = self.lock()?;
        let mut i = self.read(id)?;
        change(&mut i);
        let now = unix_now();
        if !i.status.is_alive() && i.ended.is_none() {
            i.ended = Some(rfc3339(now));
        }
        i.cost_usd = cost(&i, now);
        self.write(&i)?;
        Ok(i)
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
                Ok(i)
                    if i.ended
                        .as_deref()
                        .and_then(parse_rfc3339)
                        .is_some_and(|t| now - t > KEEP_ENDED.as_secs_f64()) =>
                {
                    self.remove(id);
                }
                Ok(mut i) => {
                    i.cost_usd = cost(&i, now);
                    out.push(i);
                }
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
        let ledger = Ledger::new(&dir.join("cloud"), dir.join("cloud").join(".lock"));
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
        assert_eq!(cost(&hour, 0.0), Some(3.6));

        ledger
            .update(&b.id, |i| i.ended = Some(rfc3339(unix_now() - KEEP_ENDED.as_secs_f64() - 60.0)))
            .unwrap();
        assert_eq!(ledger.list().len(), 1);
        assert_eq!(ledger.get(&b.id).unwrap_err().body.code, "instance_not_found", "pruned");
    }
}
