//! `DELETE /api/datasets/{id}?files=true`: deleting the folder of a dataset this server wrote, only while no run is
//! active and nothing else reads files there.

use std::collections::HashSet;
use std::path::Path;

use axum::http::StatusCode;

use crate::error::{Error, Result};
use crate::index::reads;

use super::RealStore;

impl RealStore {
    /// Unregisters the dataset and deletes the folder it was written to, the one right under `<data dir>/datasets`
    /// (whatever inside it was registered). Refused for a dataset anywhere else (`400 invalid_query`); while any run
    /// on this server is queued, running or stopping, whatever dataset it names (`409 run_active`: no trainer reads
    /// files then); and while another registered dataset, or a run of an unregistered dataset that keeps its id,
    /// reads images or labels in that folder (`409 dataset_in_use`), judged by its description as it reads now (the
    /// folders its root and sources name, resolved through links, not a cached index), or when what one of them
    /// reads cannot be checked (also `409 dataset_in_use`: nothing is deleted on a guess; one whose path is not
    /// there at all reads nothing). The checks and the
    /// deletion hold [`RealStore::files`] exclusively, so no dataset is registered and no run created or resumed in
    /// between.
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
        let runs = self.runs.entries();
        if let Some(run) = runs.iter().map(|e| e.run()).find(|r| r.status.is_active()) {
            return Err(Error::conflict(
                "run_active",
                format!("run {} is queued or training; files are deleted only while no run is", run.name),
            )
            .hint("wait for it to end, or stop it first"));
        }
        let in_use = |m: String| Error::conflict("dataset_in_use", m);
        // Whether the dataset `name` at `path` reads files in the folder; refused when that cannot be told. A
        // dataset whose folder or description is not there at all reads nothing (one moved away, or one deleted
        // with its files earlier whose runs keep its id): it must not stand in the way for good.
        let reads_folder = |name: &str, path: &str, format| {
            if matches!(Path::new(path).try_exists(), Ok(false)) {
                return Ok(false);
            }
            match reads(path, format) {
                Ok(paths) => Ok(paths.iter().any(|p| overlaps(p, &folder))),
                Err(e) => Err(in_use(format!(
                    "the files of {name} could not be checked ({e}), so they may be in that folder"
                ))
                .hint(format!("fix {name} first, or unregister this dataset without files=true"))),
            }
        };
        let others: Vec<_> = self.entries.read().unwrap().iter().filter(|e| e.id() != id).cloned().collect();
        let mut readers = Vec::new();
        for r in others.iter().map(|e| e.reg()) {
            if reads_folder(&r.name, &r.path, r.format)? {
                readers.push(r.name);
            }
        }
        if !readers.is_empty() {
            return Err(in_use(format!("{} also read files in that folder", readers.join(", ")))
                .hint("remove those datasets first, or unregister this one without files=true"));
        }
        // A run of an unregistered dataset that keeps its id finds its files again when it is registered again.
        let kept: HashSet<String> =
            self.read_registry_cached()?.tombstones.into_iter().map(|t| t.id).collect();
        for e in &runs {
            let ds = e.dataset();
            if kept.contains(&ds.id) && reads_folder(&ds.name, &ds.path, ds.format)? {
                return Err(in_use(format!(
                    "run {} of the unregistered dataset {} reads files in that folder",
                    e.run().name,
                    ds.name
                ))
                .hint("delete that run first, or unregister this dataset without files=true"));
            }
        }
        self.unregister(id)?;
        std::fs::remove_dir_all(&folder).map_err(|e| {
            let folder = folo_data::paths::shown(&folder);
            Error::internal(format!("{} was unregistered, but deleting '{folder}' failed: {e}", reg.name))
        })
    }
}

/// The resolved `p` and the canonical `folder` overlap: one holds the other.
fn overlaps(p: &Path, folder: &Path) -> bool {
    p.starts_with(folder) || folder.starts_with(p)
}
