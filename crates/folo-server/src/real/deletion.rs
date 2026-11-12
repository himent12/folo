//! `DELETE /api/datasets/{id}?files=true`: deleting the folder of a dataset this server wrote, never while anything
//! else reads files there.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use axum::http::StatusCode;

use crate::error::{Error, Result};
use crate::index::absolute;
use crate::runs::DatasetRef;
use crate::store::Store;

use super::RealStore;
use super::registry::read_registry;

impl RealStore {
    /// Unregisters the dataset and deletes the folder it was written to, the one right under `<data dir>/datasets`
    /// (whatever inside it was registered). Refused for a dataset anywhere else (`400 invalid_query`); while
    /// another registered dataset, or a run of an unregistered dataset that keeps its id, reads images or labels
    /// in that folder (`409 dataset_in_use`); and while a queued, running or stopping run does (`409
    /// run_active`). What a dataset reads is the folders its index watches (its root and split sources, by their
    /// canonical path), not only where its description lies. The checks and the deletion hold
    /// [`RealStore::files`] exclusively, so no dataset is registered and no run created or resumed in between.
    pub(super) fn delete_own_dataset(&self, id: &str) -> Result<()> {
        let _files = self.files.write().unwrap_or_else(|p| p.into_inner());
        let reg = self.entry(id)?.reg();
        let bad = |m: String| Error::new(StatusCode::BAD_REQUEST, "invalid_query", m);
        let own = std::fs::canonicalize(self.data_dir.join("datasets")).ok();
        let at = std::fs::canonicalize(&reg.path).ok();
        let folder = match (&own, &at) {
            (Some(own), Some(at)) => {
                at.strip_prefix(own).ok().and_then(|r| r.components().next()).map(|c| own.join(c))
            }
            _ => None,
        };
        let Some(folder) = folder else {
            let path = folo_data::paths::shown(&reg.path);
            return Err(bad(format!(
                "'{path}' is not a folder FOLO wrote into its data dir; its files are never deleted"
            ))
            .hint("unregister it without files=true"));
        };
        let in_use = |m: String| Error::conflict("dataset_in_use", m);
        let others: Vec<_> =
            self.entries.read().unwrap().iter().filter(|e| e.reg().id != id).cloned().collect();
        let mut readers = Vec::new();
        for e in others {
            let r = e.reg();
            // A dataset that cannot be read (its folder gone, a broken description) reads nothing now.
            let reads = match e.current() {
                Ok(idx) => idx.watched.clone(),
                Err(_) => vec![absolute(Path::new(&r.path))],
            };
            if reads.iter().any(|p| overlaps(p, &folder)) {
                readers.push(r.name);
            }
        }
        if !readers.is_empty() {
            return Err(in_use(format!("{} also read files in that folder", readers.join(", ")))
                .hint("remove those datasets first, or unregister this one without files=true"));
        }
        // Runs: every active one (whatever dataset id it names: a run reads the files its dataset path names), and
        // every run of an unregistered dataset that keeps its id, which finds its files again when it is registered
        // again.
        let kept: HashSet<String> =
            read_registry(&self.data_dir)?.tombstones.into_iter().map(|t| t.id).collect();
        let mut reads_of = std::collections::HashMap::<(String, String), bool>::new();
        for e in self.runs.entries() {
            let (run, ds) = (e.run(), e.dataset());
            let active = run.status.is_active();
            if !active && !kept.contains(&ds.id) {
                continue;
            }
            let reads = *reads_of
                .entry((ds.id.clone(), ds.path.clone()))
                .or_insert_with(|| self.dataset_reads(&ds).iter().any(|p| overlaps(p, &folder)));
            if !reads {
                continue;
            }
            if active {
                return Err(Error::conflict(
                    "run_active",
                    format!("run {} is training or queued on files in that folder", run.name),
                )
                .hint("wait for it, or stop it first"));
            }
            return Err(in_use(format!(
                "run {} of the unregistered dataset {} reads files in that folder",
                run.name, ds.name
            ))
            .hint("delete that run first, or unregister this dataset without files=true"));
        }
        self.delete_dataset(id)?;
        std::fs::remove_dir_all(&folder).map_err(|e| {
            let folder = folo_data::paths::shown(&folder);
            Error::internal(format!("{} was unregistered, but deleting '{folder}' failed: {e}", reg.name))
        })
    }

    /// The folders run dataset `ds` reads: those of the registered dataset when it is the one (same id and path),
    /// else read from its path now.
    fn dataset_reads(&self, ds: &DatasetRef) -> Vec<PathBuf> {
        if let Ok(e) = self.entry(&ds.id)
            && e.reg().path == ds.path
            && let Ok(idx) = e.current()
        {
            return idx.watched.clone();
        }
        crate::index::reads(&ds.path, ds.format)
    }
}

/// `p` and the canonical `folder` overlap: one holds the other. `p` is compared canonical when it exists.
fn overlaps(p: &Path, folder: &Path) -> bool {
    let p = std::fs::canonicalize(p).unwrap_or_else(|_| absolute(p));
    p.starts_with(folder) || folder.starts_with(&p)
}
