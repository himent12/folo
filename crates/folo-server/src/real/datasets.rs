//! Registered datasets: the registry file, their cached indexes (loaded, built and re-checked) and registering
//! a folder.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use folo_api::{Dataset, DatasetFormat};

use crate::error::{Error, Result};
use crate::index::{Fingerprint, INDEX_VERSION, Index, absolute, default_name, write_atomic};
use crate::runs::DatasetRef;
use crate::store::Store;
use crate::util::{claim_dir, fnv1a, rfc3339, unique_id, unix_now};

use super::registry::{RegistryFile, Tombstone, read_registry, registry_path, write_registry};
use super::{Entry, RealStore, Registration, State};

impl RealStore {
    pub(super) fn entry(&self, id: &str) -> Result<Arc<Entry>> {
        self.entries
            .read()
            .unwrap()
            .iter()
            .find(|e| e.reg.lock().unwrap().id == id)
            .cloned()
            .ok_or_else(|| Error::not_found("dataset", id))
    }

    /// The dataset's index, rebuilt first when its files changed (checked at most every `recheck_every`; a dataset
    /// that failed to index is retried as often).
    pub(super) fn fresh_index(&self, e: &Entry) -> Result<Arc<Index>> {
        let current = e.current();
        {
            let mut slot = e.slot.lock().unwrap();
            if slot.checking || slot.checked.elapsed() < self.recheck_every {
                return current;
            }
            slot.checking = true;
        }
        if let Ok(cur) = &current
            && Fingerprint::of(&cur.watched, &cur.excluded) == cur.fingerprint
        {
            let mut slot = e.slot.lock().unwrap();
            slot.checked = Instant::now();
            slot.checking = false;
            return current;
        }
        let reg = e.reg();
        tracing::info!(id = %reg.id, "dataset files changed; re-indexing");
        match Index::build(&reg.path, reg.format, &own_state(&self.data_dir))
            .map_err(|err| folder_gone(err, &reg))
        {
            Ok(idx) => {
                let idx = Arc::new(idx);
                save_index(&self.data_dir, e, &idx);
                self.snapshot(e, &idx);
                e.set(State::Ready(idx.clone()));
                Ok(idx)
            }
            Err(err) => {
                e.set(State::Failed(err.clone()));
                Err(err)
            }
        }
    }

    /// Updates the registry's classes / splits snapshot when they changed.
    fn snapshot(&self, e: &Entry, idx: &Index) {
        let _list = self.entries.write().unwrap();
        let changed = {
            let mut r = e.reg.lock().unwrap();
            let changed = r.classes != idx.classes || r.splits != idx.splits;
            r.classes = idx.classes.clone();
            r.splits = idx.splits;
            changed
        };
        if changed && let Ok(mut file) = read_registry(&self.data_dir) {
            let r = e.reg();
            for x in file.datasets.iter_mut().filter(|x| x.id == r.id) {
                *x = r.clone();
            }
            if let Err(err) = write_registry(&self.data_dir, &file) {
                tracing::warn!("{err}");
            }
        }
    }

    pub(super) fn meta(reg: &Registration, idx: Option<&Index>) -> Dataset {
        Dataset {
            id: reg.id.clone(),
            name: reg.name.clone(),
            // Reported with `/` separators (registrations keep the path as stored).
            path: folo_api::json::path(&reg.path),
            format: idx.map_or(reg.format, |i| i.format),
            classes: idx.map_or_else(|| reg.classes.clone(), |i| i.classes.clone()),
            created: reg.created.clone(),
            splits: idx.map_or(reg.splits, |i| i.splits),
            stats: idx.map(|i| i.stats.clone()),
        }
    }

    /// Indexes `path` and adds it to the registry.
    pub(super) fn register(
        &self,
        name: Option<String>,
        path: String,
        format: DatasetFormat,
    ) -> Result<Dataset> {
        // Shared with run admission; a deletion of dataset files waits for it (see `RealStore::files`).
        let _files = self.files.read().unwrap_or_else(|p| p.into_inner());
        // Registered absolute, so the registry works whatever the working directory of the next start.
        let path = absolute(Path::new(&path)).to_string_lossy().into_owned();
        // Registering a folder again (by its canonical path) returns the dataset it already is. Checked before
        // indexing (cheap) and again under the write lock (two requests for one folder may index at once).
        let want = std::fs::canonicalize(&path).ok();
        let same = |list: &[Arc<Entry>]| {
            let want = want.as_ref()?;
            list.iter().find_map(|e| {
                let r = e.reg();
                let same_format = format == DatasetFormat::Auto || format == r.format;
                (same_format && std::fs::canonicalize(&r.path).ok().as_ref() == Some(want)).then_some(r.id)
            })
        };
        let existing = same(&self.entries.read().unwrap());
        if let Some(id) = existing {
            return self.get_dataset(&id);
        }
        let idx = Arc::new(Index::build(&path, format, &own_state(&self.data_dir))?);
        let mut list = self.entries.write().unwrap();
        if let Some(id) = same(&list) {
            drop(list);
            return self.get_dataset(&id);
        }
        let mut file = read_registry(&self.data_dir)?;
        let used = self.run_dataset_ids();
        let live = |id: &str| list.iter().any(|e| e.reg.lock().unwrap().id == id);
        // The dataset unregistered before (its canonical path and format, from its tombstone) gets its id back, so
        // the runs that trained on it find it again. Any other gets an id no dataset, tombstone or run has.
        let back = want
            .as_deref()
            .and_then(|w| file.tombstones.iter().position(|t| t.is(w, idx.format) && !live(&t.id)));
        let id = match back {
            Some(i) => file.tombstones.remove(i).id,
            None => unique_id("ds", fnv1a(path.as_bytes()) ^ unix_now().to_bits(), |id| {
                live(id) || file.holds(id) || used.contains(id)
            }),
        };
        let name = name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| {
            let lower = |n: &str| n.to_lowercase();
            default_name(Path::new(&path), |n| {
                list.iter().any(|e| lower(&e.reg.lock().unwrap().name) == lower(n))
                    || file.datasets.iter().any(|r| lower(&r.name) == lower(n))
            })
        });
        let reg = Registration {
            id,
            name,
            path,
            format: idx.format,
            created: rfc3339(unix_now()),
            classes: idx.classes.clone(),
            splits: idx.splits,
        };
        file.datasets.push(reg.clone());
        self.save_registry(&mut file)?;
        let out = Self::meta(&reg, Some(&idx));
        let entry = Arc::new(Entry::new(reg, State::Ready(idx.clone())));
        save_index(&self.data_dir, &entry, &idx);
        list.push(entry);
        Ok(out)
    }

    /// The dataset ids the runs reference.
    fn run_dataset_ids(&self) -> HashSet<String> {
        self.runs.entries().iter().map(|e| e.dataset().id).collect()
    }

    /// Writes the registry, without the tombstones whose id no run references any more.
    pub(super) fn save_registry(&self, file: &mut RegistryFile) -> Result<()> {
        let used = self.run_dataset_ids();
        file.prune(&|id| used.contains(id));
        write_registry(&self.data_dir, file)
    }

    /// Unregisters dataset `id`, keeping its tombstone while runs reference it. The caller holds [`RealStore::files`]
    /// exclusively: a run admitted under the shared lease is in the run registry before the tombstones are pruned, so
    /// it never names an id whose identity was dropped.
    pub(super) fn unregister(&self, id: &str) -> Result<()> {
        let mut list = self.entries.write().unwrap();
        let mut file = read_registry(&self.data_dir)?;
        let entry = list.iter().find(|e| e.reg.lock().unwrap().id == id).cloned();
        let Some(reg) =
            entry.as_ref().map(|e| e.reg()).or_else(|| file.datasets.iter().find(|r| r.id == id).cloned())
        else {
            return Err(Error::not_found("dataset", id));
        };
        // Its tombstone keeps the id for it while runs reference it (`crate::real::registry`).
        file.datasets.retain(|r| r.id != id);
        file.tombstones.retain(|t| t.id != id);
        file.tombstones.push(Tombstone::of(&reg, rfc3339(unix_now())));
        self.save_registry(&mut file)?;
        list.retain(|e| e.reg.lock().unwrap().id != id);
        // Caches are ours to delete; the dataset's own files never are. Marking the entry deleted first (under the
        // lock cache writers hold) keeps in-flight indexers and thumbnails from writing them back.
        let _deleted = entry.as_ref().map(|e| {
            let mut d = e.deleted.lock().unwrap();
            *d = true;
            d
        });
        let _ = std::fs::remove_file(index_path(&self.data_dir, id));
        let _ = std::fs::remove_dir_all(self.thumbs_dir(id));
        Ok(())
    }

    pub(super) fn thumbs_dir(&self, id: &str) -> PathBuf {
        self.data_dir.join("cache").join("thumbs").join(id)
    }

    /// The registered dataset `id` as a run / eval reference.
    pub(super) fn dataset_ref(&self, e: &Entry) -> DatasetRef {
        let r = e.reg();
        DatasetRef { id: r.id, name: r.name, path: r.path, format: r.format, classes: r.classes }
    }

    /// A new, empty folder `datasets/<base>` (`-2`, `-3`, ... when taken) for a dataset this server writes.
    pub(super) fn claim_dataset_dir(&self, base: &str) -> Result<PathBuf> {
        let parent = self.data_dir.join("datasets");
        claim_dir(&parent, base)
            .map_err(|e| Error::internal(format!("creating a folder in {}: {e}", parent.display())))
    }
}

/// A registered dataset whose folder no longer exists: said as that (still `404 path_not_found`), not with the
/// registration error's advice on which folder to pass.
fn folder_gone(err: Error, reg: &Registration) -> Error {
    if err.body.code != "path_not_found" || Path::new(&reg.path).exists() {
        return err;
    }
    let (name, path) = (&reg.name, folo_data::paths::shown(&reg.path));
    let message = format!("the folder of dataset '{name}' is gone: '{path}' does not exist");
    Error::new(StatusCode::NOT_FOUND, "path_not_found", message).hint(format!(
        "move the folder back (FOLO picks it up again within seconds), or unregister the dataset with \
         DELETE /api/datasets/{} (its files are never deleted)",
        reg.id
    ))
}

/// The server's own files in the data dir, left out of dataset fingerprints (the data dir may sit inside a
/// registered dataset folder; caching an index must not make it stale).
fn own_state(dir: &Path) -> Vec<PathBuf> {
    vec![dir.join("cache"), dir.join("label"), dir.join("bench"), registry_path(dir)]
}

pub(super) fn index_path(dir: &Path, id: &str) -> PathBuf {
    dir.join("cache").join("datasets").join(format!("{id}.json"))
}

fn save_index(dir: &Path, e: &Entry, idx: &Index) {
    // An index built under a read guard may be partial (refused files skipped); never let it stand in for the
    // ordinary cache that an unguarded process would trust.
    if folo_data::guard::installed() {
        return;
    }
    let id = e.reg().id;
    let path = index_path(dir, &id);
    let bytes = serde_json::to_vec(idx).expect("indexes serialize");
    e.write_cache(|| {
        if let Err(err) = write_atomic(&path, &bytes) {
            tracing::warn!("caching the index of {id} in {}: {err}", path.display());
        }
    });
}

/// Startup: loads the cached index when it is fresh, rebuilds it otherwise.
pub(super) fn load_or_build(dir: &Path, e: &Entry) {
    let reg = e.reg();
    let cached = std::fs::read(index_path(dir, &reg.id))
        .ok()
        .and_then(|b| serde_json::from_slice::<Index>(&b).ok())
        .filter(|i| i.version == INDEX_VERSION)
        // A cache built without the read guard may hold data from files the guard refuses: rebuild under it.
        .filter(|_| !folo_data::guard::installed());
    if let Some(idx) = cached
        && idx.is_fresh()
    {
        tracing::debug!(id = %reg.id, "loaded the cached index");
        e.set(State::Ready(Arc::new(idx)));
        return;
    }
    let t0 = Instant::now();
    match Index::build(&reg.path, reg.format, &own_state(dir)).map_err(|err| folder_gone(err, &reg)) {
        Ok(idx) => {
            tracing::info!(id = %reg.id, images = idx.images.len(), secs = t0.elapsed().as_secs_f32(), "indexed");
            save_index(dir, e, &idx);
            e.set(State::Ready(Arc::new(idx)));
        }
        Err(err) => {
            tracing::warn!(id = %reg.id, path = %reg.path, "indexing failed: {err}");
            e.set(State::Failed(err));
        }
    }
}
