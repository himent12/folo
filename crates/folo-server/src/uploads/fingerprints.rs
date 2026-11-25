//! Remembered upload fingerprints (`<data dir>/uploads.json`), cached in memory after first read and updated
//! atomically on write.

use std::path::PathBuf;
use std::sync::Mutex;

use folo_api::UploadKind;
use serde::{Deserialize, Serialize};

use crate::index::write_atomic;

/// One finished upload with a fingerprint (`uploads.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Known {
    pub(super) kind: UploadKind,
    pub(super) fingerprint: String,
    /// The dataset id or model id it became.
    pub(super) id: String,
}

/// `<data dir>/uploads.json`, cached in memory after its first read and updated atomically on write. A missing or
/// malformed file remembers nothing (the next upload sends its content).
pub(super) struct Fingerprints {
    path: PathBuf,
    cache: Mutex<Option<Vec<Known>>>,
    #[cfg(test)]
    pub(super) disk_reads: std::sync::atomic::AtomicUsize,
}

impl Fingerprints {
    pub(super) fn new(path: PathBuf) -> Fingerprints {
        Fingerprints {
            path,
            cache: Mutex::default(),
            #[cfg(test)]
            disk_reads: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Removes stale `.uploads.json.<pid>.<seq>.tmp` files left in the parent folder by a crashed process.
    pub(super) fn cleanup_stale_temps(&self) {
        let Some(parent) = self.path.parent() else { return };
        let Some(name) = self.path.file_name().and_then(|n| n.to_str()) else { return };
        let prefix = format!(".{name}.");
        let Ok(entries) = std::fs::read_dir(parent) else { return };
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let Some(s) = file_name.to_str() else { continue };
            if is_atomic_temp_name(s, &prefix)
                && let Err(e) = std::fs::remove_file(entry.path())
                && e.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!("removing stale upload temp file {}: {e}", entry.path().display());
            }
        }
    }

    /// Reads `uploads.json` from disk: `Some(Vec::new())` when missing or malformed (safe to cache/replace),
    /// `None` on transient I/O errors so callers retry later and never overwrite from an unknown baseline.
    fn read_disk(&self) -> Option<Vec<Known>> {
        #[cfg(test)]
        self.disk_reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let bytes = match std::fs::read(&self.path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(Vec::new()),
            Err(e) => {
                tracing::warn!("reading {}: {e}", self.path.display());
                return None;
            }
        };
        Some(serde_json::from_slice(&bytes).unwrap_or_else(|e| {
            tracing::warn!(
                "{} is malformed ({e}); it is replaced on the next finished upload",
                self.path.display()
            );
            Vec::new()
        }))
    }

    fn load<'a>(&self, cache: &'a mut Option<Vec<Known>>) -> Option<&'a mut Vec<Known>> {
        if cache.is_none() {
            *cache = self.read_disk();
        }
        cache.as_mut()
    }

    pub(super) fn find(&self, kind: UploadKind, fingerprint: &str) -> Option<String> {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let list = self.load(&mut cache)?;
        list.iter().find(|k| k.kind == kind && k.fingerprint == fingerprint).map(|k| k.id.clone())
    }

    /// Records `k`, replacing an older record of its kind and fingerprint.
    pub(super) fn write(&self, k: Known) {
        let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
        let Some(list) = self.load(&mut cache) else {
            return;
        };
        let mut next = list.clone();
        next.retain(|x| x.kind != k.kind || x.fingerprint != k.fingerprint);
        next.push(k);
        let bytes = serde_json::to_vec_pretty(&next).expect("fingerprints serialize");
        match write_atomic(&self.path, &bytes) {
            Ok(()) => *list = next,
            Err(e) => tracing::warn!("writing {}: {e}", self.path.display()),
        }
    }
}

/// True iff `name` matches `<prefix><pid>.<seq>.tmp` produced by [`write_atomic`].
fn is_atomic_temp_name(name: &str, prefix: &str) -> bool {
    let Some(mid) = name.strip_prefix(prefix).and_then(|s| s.strip_suffix(".tmp")) else {
        return false;
    };
    let Some((pid, seq)) = mid.split_once('.') else {
        return false;
    };
    !pid.is_empty()
        && !seq.is_empty()
        && pid.bytes().all(|b| b.is_ascii_digit())
        && seq.bytes().all(|b| b.is_ascii_digit())
}
