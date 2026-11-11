//! Registered datasets: the registry file, their cached indexes (loaded, built and re-checked) and registering
//! a folder.

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
        if changed && let Ok(mut regs) = read_registry(&self.data_dir) {
            let r = e.reg();
            for x in regs.iter_mut().filter(|x| x.id == r.id) {
                *x = r.clone();
            }
            if let Err(err) = write_registry(&self.data_dir, &regs) {
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
        let mut regs = read_registry(&self.data_dir)?;
        // Unique against loaded datasets and registrations whose folder is currently missing. Derived from the
        // folder (canonical, so any spelling) and its format, not the time: a folder unregistered and registered
        // again gets its id back, and the runs that trained on it find their dataset again.
        let id =
            unique_id("ds", dataset_seed(want.as_deref().unwrap_or(Path::new(&path)), idx.format), |id| {
                list.iter().any(|e| e.reg.lock().unwrap().id == id) || regs.iter().any(|r| r.id == id)
            });
        let name = name.filter(|n| !n.trim().is_empty()).unwrap_or_else(|| {
            let lower = |n: &str| n.to_lowercase();
            default_name(Path::new(&path), |n| {
                list.iter().any(|e| lower(&e.reg.lock().unwrap().name) == lower(n))
                    || regs.iter().any(|r| lower(&r.name) == lower(n))
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
        regs.push(reg.clone());
        write_registry(&self.data_dir, &regs)?;
        let out = Self::meta(&reg, Some(&idx));
        let entry = Arc::new(Entry::new(reg, State::Ready(idx.clone())));
        save_index(&self.data_dir, &entry, &idx);
        list.push(entry);
        Ok(out)
    }

    pub(super) fn thumbs_dir(&self, id: &str) -> PathBuf {
        self.data_dir.join("cache").join("thumbs").join(id)
    }

    /// The registered dataset `id` as a run / eval reference.
    pub(super) fn dataset_ref(&self, e: &Entry) -> DatasetRef {
        let r = e.reg();
        DatasetRef { id: r.id, name: r.name, path: r.path, format: r.format }
    }

    /// A new, empty folder `datasets/<base>` (`-2`, `-3`, ... when taken) for a dataset this server writes.
    pub(super) fn claim_dataset_dir(&self, base: &str) -> Result<PathBuf> {
        let parent = self.data_dir.join("datasets");
        claim_dir(&parent, base)
            .map_err(|e| Error::internal(format!("creating a folder in {}: {e}", parent.display())))
    }
}

/// The seed of a registered dataset's id: its canonical path and the format it is read in.
pub(crate) fn dataset_seed(canonical: &Path, format: DatasetFormat) -> u64 {
    fnv1a(canonical.to_string_lossy().as_bytes()) ^ fnv1a(format!("{format:?}").as_bytes())
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

fn registry_path(dir: &Path) -> PathBuf {
    dir.join("datasets.json")
}

pub(super) fn index_path(dir: &Path, id: &str) -> PathBuf {
    dir.join("cache").join("datasets").join(format!("{id}.json"))
}

/// The registry. A missing file is an empty registry; an unreadable or malformed one is an error, so it is never
/// overwritten (and its registrations lost) by the next write.
pub(super) fn read_registry(dir: &Path) -> Result<Vec<Registration>> {
    let path = registry_path(dir);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(Error::internal(format!("reading {}: {e}", path.display()))),
    };
    serde_json::from_str(&text).map_err(|e| {
        Error::internal(format!("{} is malformed: {e}", path.display()))
            .hint("fix the JSON by hand or move the file away; FOLO will not overwrite it")
    })
}

pub(super) fn write_registry(dir: &Path, regs: &[Registration]) -> Result<()> {
    let path = registry_path(dir);
    let bytes = serde_json::to_vec_pretty(regs).expect("registrations serialize");
    write_atomic(&path, &bytes).map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))
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
