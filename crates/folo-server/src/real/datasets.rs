//! Registered datasets: the registry file, their cached indexes (loaded, built and re-checked) and registering
//! a folder.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::http::StatusCode;
use folo_api::{Dataset, DatasetFormat};

use crate::error::{Error, Result};
use crate::index::{Fingerprint, INDEX_VERSION, Index, default_name, free_name, write_atomic};
use crate::runs::DatasetRef;
use crate::store::Store;
use crate::util::{claim_dir, fnv1a, rfc3339, unique_id, unix_now};

use super::registry::{Gains, RegistryFile, Tombstone, read_registry, registry_path, tombstones_path};
use super::{Entry, EntryTable, RealStore, Registration, State};

/// Resets [`Slot::checking`] if an index re-check unwinds before reaching [`Entry::set`].
struct CheckingGuard<'a> {
    entry: &'a Entry,
    armed: bool,
}

impl<'a> CheckingGuard<'a> {
    fn new(entry: &'a Entry) -> Self {
        Self { entry, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for CheckingGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            let mut slot = self.entry.slot.lock().unwrap_or_else(|p| p.into_inner());
            slot.checking = false;
        }
    }
}

/// What a dataset being registered is called.
pub(super) enum Naming {
    /// Its folder's name (`default_name`), numbered while a dataset has it.
    Default,
    /// The name asked for, as it is (a registration's own `name`, an upload's).
    Given(String),
    /// The name asked for, numbered `-2`, `-3`, ... while a dataset has it (a generated sample set: two generations
    /// of one name, also at once, are told apart).
    Unique(String),
}

/// Generates a unique, unguessable per-registration rollback token (`<seq>-<random hex>`).
fn new_rollback_token() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seq = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut rnd = [0u8; 12];
    let _ = getrandom::fill(&mut rnd);
    let mut out = format!("{seq:x}-");
    for b in rnd {
        use std::fmt::Write;
        let _ = write!(&mut out, "{b:02x}");
    }
    out
}

impl RealStore {
    pub(super) fn entry(&self, id: &str) -> Result<Arc<Entry>> {
        self.entries.read().unwrap().get(id).ok_or_else(|| Error::not_found("dataset", id))
    }

    /// Reads the dataset registry through the stat-validated in-memory cache.
    pub(super) fn read_registry_cached(&self) -> Result<RegistryFile> {
        self.registry.lock().unwrap_or_else(|p| p.into_inner()).read(&self.data_dir)
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
        let mut guard = CheckingGuard::new(e);
        if let Ok(cur) = &current
            && Fingerprint::of(&cur.watched, &cur.excluded) == cur.fingerprint
        {
            let mut slot = e.slot.lock().unwrap();
            slot.checked = Instant::now();
            slot.checking = false;
            guard.disarm();
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
                guard.disarm();
                Ok(idx)
            }
            Err(err) => {
                e.set(State::Failed(err.clone()));
                guard.disarm();
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
        if changed && let Ok(mut file) = self.read_registry_cached() {
            let r = e.reg();
            for x in file.datasets.iter_mut().filter(|x| x.id == r.id) {
                *x = r.clone();
            }
            if let Err(err) = self.registry.lock().unwrap_or_else(|p| p.into_inner()).write(
                &self.data_dir,
                &mut file,
                Gains::Registration,
            ) {
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
            // The files' fingerprint: a label edited in place changes it, whatever it did to the stats.
            revision: idx.map(|i| i.fingerprint.hash.clone()),
        }
    }

    /// `Err` while `datasets.json`, unreadable when this server started, lists datasets: they were not loaded, so a
    /// list would leave them out and registering one again could give it a second id. Said (with the file's own
    /// error while it is still damaged) to the list, registering and unregistering until a restart loads them; a
    /// registry fixed to list none lifts it (x3-server-7, review rounds 1 and 2).
    pub(super) fn registry_loaded(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        if !self.registry_unread.load(Ordering::SeqCst) {
            return Ok(());
        }
        let n = read_registry(&self.data_dir)?.datasets.len();
        if n == 0 {
            self.registry_unread.store(false, Ordering::SeqCst);
            return Ok(());
        }
        let (path, s) = (registry_path(&self.data_dir), if n == 1 { "" } else { "s" });
        Err(Error::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "internal",
            format!("{} lists {n} dataset{s} this server did not load when it started", path.display()),
        )
        .hint("restart folo serve to load them"))
    }

    /// Indexes `path` and adds it to the registry.
    pub(super) fn register(&self, name: Naming, path: String, format: DatasetFormat) -> Result<Dataset> {
        self.register_tracked(name, path, format, false).map(|(ds, _)| ds)
    }

    /// Indexes `path` and adds it to the registry, returning `(dataset, rollback_token)` where `rollback_token` is
    /// `Some` iff `exclusive` is `true` and this call inserted a brand-new registration (rather than returning an
    /// existing one).
    pub(super) fn register_tracked(
        &self,
        name: Naming,
        path: String,
        format: DatasetFormat,
        exclusive: bool,
    ) -> Result<(Dataset, Option<String>)> {
        self.registry_loaded()?;
        // Registered absolute, so the registry works whatever the working directory of the next start.
        let path = crate::index::resolve_input_path(&path).to_string_lossy().into_owned();
        // Registering a folder again (by its canonical path) returns the dataset it already is. Checked before
        // indexing (cheap) and again under the write lock (two requests for one folder may index at once).
        let want = std::fs::canonicalize(&path).ok();
        let same = |table: &EntryTable| {
            let want = want.as_ref()?;
            table.iter().find_map(|e| {
                let r_path = {
                    let r = e.reg.lock().unwrap();
                    if format != DatasetFormat::Auto && format != r.format {
                        return None;
                    }
                    r.path.clone()
                };
                (std::fs::canonicalize(&r_path).ok().as_ref() == Some(want)).then(|| e.clone())
            })
        };
        let existing = {
            let table = self.entries.read().unwrap();
            same(&table).map(|e| {
                // Mark shared while holding the entries lock so a concurrent `unregister_if_unused` cannot race past.
                e.shared.store(true, std::sync::atomic::Ordering::SeqCst);
                e.id().to_string()
            })
        };
        if let Some(id) = existing {
            return self.get_dataset(&id).map(|ds| (ds, None));
        }
        // A path that is not there or not a dataset is said before the lease (and without walking it), so it never
        // holds up a dataset deletion (x3-server-1).
        crate::index::preflight(&path, format)?;
        // Shared with run admission; a deletion of dataset files waits for it (see `RealStore::files`).
        let _files = self.files.read().unwrap_or_else(|p| p.into_inner());
        let idx = Arc::new(Index::build(&path, format, &own_state(&self.data_dir))?);
        let mut table = self.entries.write().unwrap();
        if let Some(e) = same(&table) {
            e.shared.store(true, std::sync::atomic::Ordering::SeqCst);
            let id = e.id().to_string();
            drop(table);
            return self.get_dataset(&id).map(|ds| (ds, None));
        }
        let mut file = self.read_registry_cached()?;
        let used = self.run_dataset_ids();
        let live = |id: &str| table.contains(id);
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
        // Under the registry's write lock, so two registrations at once never both take a free name.
        let lower = |n: &str| n.to_lowercase();
        let taken = |n: &str| {
            table.iter().any(|e| lower(&e.reg.lock().unwrap().name) == lower(n))
                || file.datasets.iter().any(|r| lower(&r.name) == lower(n))
        };
        let name = match name {
            Naming::Given(n) if !n.trim().is_empty() => n,
            Naming::Unique(n) if !n.trim().is_empty() => free_name(n, taken),
            _ => default_name(Path::new(&path), taken),
        };
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
        self.save_registry(&mut file, Gains::Registration)?;
        let out = Self::meta(&reg, Some(&idx));
        let rollback = exclusive.then(new_rollback_token);
        let entry = Arc::new(Entry::with_rollback(reg, State::Ready(idx.clone()), rollback.clone()));
        save_index(&self.data_dir, &entry, &idx);
        table.insert(entry);
        Ok((out, rollback))
    }

    /// The dataset ids the runs reference.
    fn run_dataset_ids(&self) -> HashSet<String> {
        self.runs.entries().iter().map(|e| e.dataset().id).collect()
    }

    /// Writes the registry, without the tombstones whose id no run references any more.
    pub(super) fn save_registry(&self, file: &mut RegistryFile, gains: Gains) -> Result<()> {
        let used = self.run_dataset_ids();
        file.prune(&|id| used.contains(id));
        self.registry.lock().unwrap_or_else(|p| p.into_inner()).write(&self.data_dir, file, gains)
    }

    /// Unregisters dataset `id`, keeping its tombstone while runs reference it. The caller holds [`RealStore::files`]
    /// exclusively: a run admitted under the shared lease is in the run registry before the tombstones are pruned, so
    /// it never names an id whose identity was dropped.
    pub(super) fn unregister(&self, id: &str) -> Result<()> {
        self.registry_loaded()?;
        let table = self.entries.write().unwrap();
        self.unregister_locked(id, table)
    }

    /// Unregisters `id` only when its active registration matches `token`, has not been shared with any second
    /// registration, and is not referenced by any run or benchmark (persisted or in-flight). Atomically checks and
    /// unregisters under [`RealStore::files`] and [`RealStore::entries`] write locks.
    pub(super) fn unregister_if_unused(&self, id: &str, token: &str) -> Result<()> {
        let _files = self.files.write().unwrap_or_else(|p| p.into_inner());
        self.registry_loaded()?;
        let table = self.entries.write().unwrap();
        let Some(entry) = table.get(id) else {
            return Err(Error::not_found("dataset", id));
        };
        if entry.rollback_token.as_deref() != Some(token) {
            return Err(Error::conflict(
                "dataset_in_use",
                format!("dataset '{id}' registration does not match rollback token"),
            ));
        }
        if entry.shared.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Error::conflict(
                "dataset_in_use",
                format!("dataset '{id}' is shared by another caller"),
            ));
        }
        if self.runs.entries().iter().any(|e| e.dataset().id == id) {
            return Err(Error::conflict(
                "dataset_in_use",
                format!("dataset '{id}' is referenced by a training run"),
            ));
        }
        if self.benches.as_ref().is_some_and(|b| b.references_dataset(id)) {
            return Err(Error::conflict(
                "dataset_in_use",
                format!("dataset '{id}' is referenced by a benchmark"),
            ));
        }
        self.unregister_locked(id, table)
    }

    fn unregister_locked(
        &self,
        id: &str,
        mut table: std::sync::RwLockWriteGuard<'_, EntryTable>,
    ) -> Result<()> {
        let mut file = self.read_registry_cached()?;
        let entry = table.get(id);
        let Some(reg) =
            entry.as_ref().map(|e| e.reg()).or_else(|| file.datasets.iter().find(|r| r.id == id).cloned())
        else {
            return Err(Error::not_found("dataset", id));
        };
        // Its tombstone keeps the id for it while runs reference it (`crate::real::registry`).
        file.datasets.retain(|r| r.id != id);
        file.tombstones.retain(|t| t.id != id);
        file.tombstones.push(Tombstone::of(&reg, rfc3339(unix_now())));
        self.save_registry(&mut file, Gains::Tombstone)?;
        table.remove(id);
        // Caches are ours to delete; the dataset's own files never are. Marking the entry deleted first (under the
        // lock cache writers hold) keeps in-flight indexers and thumbnails from writing them back.
        let _deleted = entry.as_ref().map(|e| {
            let mut d = e.deleted.lock().unwrap();
            *d = true;
            d
        });
        self.thumbs.lock().unwrap_or_else(|p| p.into_inner()).evict_dataset(id);
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
    vec![dir.join("cache"), dir.join("label"), dir.join("bench"), registry_path(dir), tombstones_path(dir)]
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
    let id = e.id();
    let path = index_path(dir, id);
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
