//! Models of the real store: the weights of finished runs, the `.folo` files under `<data dir>/models` and the
//! published weights downloaded with `folo weights pull` ([`folo_api::weights`]), their metadata (read once per file
//! version), validation results (`/eval`, computed with folo-vision's [`validate`](folo_vision::val::validate) and
//! cached in memory and under `cache/eval/`) and exports.
//!
//! Files:
//! - `mod.rs`: [`Models`], bounded LRU caches ([`BoundedCache`]), `<data dir>/models` scanning and published weights.
//! - `meta.rs`: [`FileMeta`], [`ModelRef`], `.folo` header parsing ([`read_meta`]), FLOPs and model ID helpers.
//! - `weights.rs`: cached derived weights (`cache/weights/` for `f16` and `onnx`) with per-`(model, variant)` locks.
//! - `eval.rs`: single-flight cached `/eval` computation, `POST /api/val`, and [`eval_of`].
//! - `export.rs`: `POST /api/models/{id}/export` publishing to `<data dir>/exports/` and [`onnx_export`].
//! - `tests.rs`: unit tests for metadata parsing, bounded caching, single-flight eval, exports, and derived weights.

mod eval;
mod export;
mod meta;
mod weights;

#[cfg(test)]
mod tests;

use std::borrow::Borrow;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

use folo_api::ModelSource;

use crate::engine::FileKey;
use crate::error::{Error, Result};

pub use eval::{EVAL_BATCH, check_val_params, val_with_diagnostics};
pub use export::onnx_export;
pub use meta::{FileMeta, ModelRef, file_model_id, model_of};
pub(crate) use meta::{gflops_640, read_meta};

use eval::EvalCache;
use weights::DerivedLocks;

/// Maximum `.folo` header metadata entries kept in memory.
pub(crate) const MAX_META_CACHE: usize = 4096;
/// Maximum unreadable-file warning suppressions kept in memory.
pub(crate) const MAX_SKIPPED_CACHE: usize = 512;
/// Maximum file-model `id -> path` index entries kept in memory.
pub(crate) const MAX_FILE_ID_CACHE: usize = 4096;
/// Maximum `/eval` reports kept in memory (each report holds PR curves and a confusion matrix).
pub(crate) const MAX_EVAL_CACHE: usize = 128;

/// Small bounded map with exact LRU eviction so long-running server caches never grow without bound.
pub(crate) struct BoundedCache<K, V> {
    cap: usize,
    clock: u64,
    entries: HashMap<K, (u64, V)>,
}

impl<K: Eq + Hash, V> BoundedCache<K, V> {
    pub(crate) fn new(cap: usize) -> Self {
        Self { cap, clock: 0, entries: HashMap::new() }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1);
        self.clock
    }

    pub(crate) fn get<Q>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let stamp = self.tick();
        self.entries.get_mut(key).map(|(last, val)| {
            *last = stamp;
            &*val
        })
    }

    pub(crate) fn insert(&mut self, key: K, val: V) -> Option<V>
    where
        K: Clone,
    {
        if self.cap == 0 {
            return None;
        }
        let stamp = self.tick();
        if let Some((last, existing)) = self.entries.get_mut(&key) {
            *last = stamp;
            return Some(std::mem::replace(existing, val));
        }
        if self.entries.len() >= self.cap
            && let Some(oldest) = self.entries.iter().min_by_key(|(_, (t, _))| *t).map(|(k, _)| k.clone())
        {
            self.entries.remove(&oldest);
        }
        self.entries.insert(key, (stamp, val));
        None
    }

    pub(crate) fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.entries.remove(key).map(|(_, v)| v)
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    pub(crate) fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.entries.contains_key(key)
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Returns whether `path` is a regular `.folo` file strictly under `models_dir` with no symlinked file or
/// intermediate directory — matching [`Models::collect_model_files`] (`WalkDir` with `follow_links(false)`).
fn is_regular_model_file(models_dir: &Path, path: &Path) -> bool {
    if !path.extension().is_some_and(|x| x.eq_ignore_ascii_case("folo")) {
        return false;
    }
    let Ok(rel) = path.strip_prefix(models_dir) else {
        return false;
    };
    let mut components = rel.components().peekable();
    if components.peek().is_none() {
        return false;
    }
    let mut cur = models_dir.to_path_buf();
    while let Some(comp) = components.next() {
        let Component::Normal(part) = comp else {
            return false;
        };
        cur.push(part);
        let Ok(meta) = std::fs::symlink_metadata(&cur) else {
            return false;
        };
        let ft = meta.file_type();
        if ft.is_symlink() {
            return false;
        }
        if components.peek().is_some() {
            if !ft.is_dir() {
                return false;
            }
        } else if !ft.is_file() {
            return false;
        }
    }
    true
}

/// Metadata, eval, file-id and derived-weights caches.
pub struct Models {
    data_dir: PathBuf,
    /// The `folo weights` cache whose downloaded files list as published models (`None`: none are listed).
    published_dir: Option<PathBuf>,
    meta: Mutex<BoundedCache<PathBuf, (FileKey, Arc<FileMeta>)>>,
    evals: Mutex<EvalCache>,
    /// The weights files listed as skipped (unreadable), each with the version that was: warned about once.
    skipped: Mutex<BoundedCache<PathBuf, FileKey>>,
    /// Bounded `model_id -> path` index for `<data dir>/models` files so `GET /api/models/{id}` avoids walking the
    /// folder on warm lookups while still validating `FileKey` against disk.
    by_id: Mutex<BoundedCache<String, PathBuf>>,
    /// Per-`(model, variant)` locks for derived weights (f16, ONNX), so concurrent exports of distinct models or
    /// variants run in parallel while duplicate first requests export once.
    derived: Mutex<DerivedLocks>,
}

impl Models {
    /// Caches rooted at `data_dir`.
    pub fn new(data_dir: PathBuf) -> Models {
        Models::with_caps(data_dir, MAX_META_CACHE, MAX_SKIPPED_CACHE, MAX_FILE_ID_CACHE, MAX_EVAL_CACHE)
    }

    pub(crate) fn with_caps(
        data_dir: PathBuf,
        meta_cap: usize,
        skipped_cap: usize,
        file_id_cap: usize,
        eval_cap: usize,
    ) -> Models {
        Models {
            data_dir,
            published_dir: None,
            meta: Mutex::new(BoundedCache::new(meta_cap)),
            evals: Mutex::new(EvalCache::new(eval_cap)),
            skipped: Mutex::new(BoundedCache::new(skipped_cap)),
            by_id: Mutex::new(BoundedCache::new(file_id_cap)),
            derived: Mutex::default(),
        }
    }

    /// Says in the log that the weights file at `path` is not listed, why, and only once per version of the file:
    /// the models are listed again on every `GET /api/models`, which both apps poll (x3-server-8). Returns whether
    /// it was said now.
    pub fn warn_skipped(&self, path: &Path, err: &Error) -> bool {
        let Ok(key) = FileKey::of(path) else { return false };
        let mut skipped = self.skipped.lock().unwrap_or_else(|p| p.into_inner());
        if skipped.insert(path.to_path_buf(), key.clone()).as_ref() == Some(&key) {
            return false;
        }
        drop(skipped);
        tracing::warn!("skipping {}: {err}", path.display());
        true
    }

    /// Lists the published weights downloaded into `dir` (the `folo weights` cache, [`folo_api::weights::cache_dir`])
    /// as models.
    pub fn set_published(&mut self, dir: Option<PathBuf>) {
        self.published_dir = dir;
    }

    /// The `folo weights` cache this store lists published models from.
    pub fn published_dir(&self) -> Option<&Path> {
        self.published_dir.as_deref()
    }

    /// One downloaded published model `p` from `dir`, or `None` when not downloaded or unreadable.
    fn published_entry(&self, dir: &Path, p: &folo_api::weights::Published) -> Option<ModelRef> {
        let path = dir.join(p.asset);
        if !path.is_file() {
            return None;
        }
        match self.meta(&path) {
            Ok(meta) => {
                let mut model = model_of(p.name.to_string(), p.name.to_string(), &meta, None, meta.metrics);
                model.source = Some(ModelSource::Published);
                model.about = Some(p.about.to_string());
                Some(ModelRef::new(model, path, None))
            }
            Err(e) => {
                self.warn_skipped(&path, &e);
                None
            }
        }
    }

    /// The published weights downloaded into the cache, as models: id and name are the published name (what
    /// `folo weights pull` and `train --weights` take), `source: published`, `about` from the registry.
    pub fn published(&self) -> Vec<ModelRef> {
        let Some(dir) = &self.published_dir else { return Vec::new() };
        folo_api::weights::PUBLISHED.iter().filter_map(|p| self.published_entry(dir, p)).collect()
    }

    /// The published model `name`, when it is downloaded (checks only that asset, without scanning the rest).
    pub fn published_model(&self, name: &str) -> Option<ModelRef> {
        let dir = self.published_dir.as_ref()?;
        let p = folo_api::weights::find(name)?;
        self.published_entry(dir, p)
    }

    /// The folder scanned for `.folo` files.
    pub fn models_dir(&self) -> PathBuf {
        self.data_dir.join("models")
    }

    /// Reads metadata for `path`, serving from cache when `FileKey` matches and optionally caching on a miss.
    fn meta_inner(&self, path: &Path, insert_on_miss: bool) -> Result<Arc<FileMeta>> {
        let key = FileKey::of(path).map_err(|e| Error::internal(format!("{}: {e}", path.display())))?;
        if let Some((k, m)) = self.meta.lock().unwrap_or_else(|p| p.into_inner()).get(path)
            && *k == key
        {
            return Ok(m.clone());
        }
        let m = Arc::new(read_meta(path)?);
        let mut guard = self.meta.lock().unwrap_or_else(|p| p.into_inner());
        if insert_on_miss || guard.contains_key(path) {
            guard.insert(path.to_path_buf(), (key, m.clone()));
        }
        Ok(m)
    }

    /// The metadata of `path`, read again only when the file changed.
    pub fn meta(&self, path: &Path) -> Result<Arc<FileMeta>> {
        self.meta_inner(path, true)
    }

    /// Builds a [`ModelRef`] for a `.folo` file under `<data dir>/models`.
    fn file_model_at(&self, path: PathBuf, insert_meta: bool, insert_by_id: bool) -> Option<ModelRef> {
        match self.meta_inner(&path, insert_meta) {
            Ok(meta) => {
                let id = file_model_id(&path);
                let name =
                    path.file_stem().map_or_else(|| "model".into(), |s| s.to_string_lossy().into_owned());
                let model = model_of(id.clone(), name, &meta, None, meta.metrics);
                if insert_by_id {
                    self.by_id.lock().unwrap_or_else(|p| p.into_inner()).insert(id, path.clone());
                }
                Some(ModelRef::new(model, path, None))
            }
            Err(e) => {
                self.warn_skipped(&path, &e);
                None
            }
        }
    }

    /// Collects every regular `.folo` file under `<data dir>/models` in deterministic sorted order without
    /// following symlinks.
    fn collect_model_files(&self) -> Vec<PathBuf> {
        let dir = self.models_dir();
        if !dir.is_dir() {
            return Vec::new();
        }
        let mut files: Vec<PathBuf> = walkdir::WalkDir::new(&dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("folo")))
            .collect();
        files.sort();
        files
    }

    /// One `.folo` file model by `id`: checks the bounded `by_id` index first (verifying the path is still a regular
    /// `.folo` file inside `<data dir>/models` without traversing symlinks, and checking `FileKey` via
    /// [`Models::meta`]), falling back to a path-only walk that reads and warms only the matching file.
    pub fn file_model(&self, id: &str) -> Option<ModelRef> {
        let dir = self.models_dir();
        let cached = self.by_id.lock().unwrap_or_else(|p| p.into_inner()).get(id).cloned();
        if let Some(path) = cached {
            if is_regular_model_file(&dir, &path) && file_model_id(&path) == id {
                if let Some(m) = self.file_model_at(path.clone(), true, true) {
                    return Some(m);
                }
            } else {
                self.meta.lock().unwrap_or_else(|p| p.into_inner()).remove(&path);
                self.skipped.lock().unwrap_or_else(|p| p.into_inner()).remove(&path);
            }
            self.by_id.lock().unwrap_or_else(|p| p.into_inner()).remove(id);
        }
        for path in self.collect_model_files() {
            if file_model_id(&path) != id {
                continue;
            }
            if let Some(m) = self.file_model_at(path, true, true) {
                return Some(m);
            }
        }
        None
    }

    /// Every `.folo` file under the models folder (recursively), as models.
    pub fn scan(&self) -> Vec<ModelRef> {
        let files = self.collect_model_files();
        let mut meta_budget = self.meta.lock().unwrap_or_else(|p| p.into_inner()).cap();
        let mut id_budget = self.by_id.lock().unwrap_or_else(|p| p.into_inner()).cap();
        let mut seen_ids = HashSet::new();
        let mut out = Vec::with_capacity(files.len());
        for path in files {
            let allow_meta = meta_budget > 0;
            let Some(m) = self.file_model_at(path, allow_meta, false) else { continue };
            if allow_meta {
                meta_budget -= 1;
            }
            // First valid path in sorted order wins on a 32-bit id collision, matching `file_model` cold lookups;
            // cap insertions per scan so catalogs larger than the cache keep earlier entries warm instead of
            // thrashing every consecutive listing.
            if seen_ids.insert(m.model.id.clone()) {
                let mut by_id = self.by_id.lock().unwrap_or_else(|p| p.into_inner());
                if id_budget > 0 {
                    by_id.insert(m.model.id.clone(), m.path.clone());
                    id_budget -= 1;
                } else if by_id.contains_key(&m.model.id) {
                    by_id.insert(m.model.id.clone(), m.path.clone());
                }
            }
            out.push(m);
        }
        out
    }
}
