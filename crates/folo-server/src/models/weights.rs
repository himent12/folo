//! Cached derived weights (`cache/weights/` for `f16` and `onnx`) with per-`(model, variant)` locking and
//! version-scoped stale-file pruning.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock, Weak};

use crate::engine::{self, FileKey, engine_error};
use crate::error::{Error, Result};
use crate::store::{WeightsFormat, weights_io};
use crate::util::fnv1a;

use super::export::{export_folo_f16, onnx_export};
use super::{ModelRef, Models};

/// Per-model and per-`(model, variant)` locks for `cache/weights/`:
/// - Different models never block each other.
/// - Different variants of one model (e.g. `f16.folo` vs `onnx`) export concurrently under a shared model read
///   lock, while duplicate requests for the same `(model, variant)` serialize on its variant mutex.
/// - [`Models::forget_weights`] takes that model's write lock so it waits for every in-flight export of that
///   model without blocking exports of other models.
#[derive(Default)]
pub(super) struct DerivedLocks {
    pub(super) models: HashMap<String, Weak<RwLock<()>>>,
    pub(super) variants: HashMap<(String, String), Weak<Mutex<()>>>,
}

impl DerivedLocks {
    pub(super) fn model(&mut self, id: &str) -> Arc<RwLock<()>> {
        if let Some(lock) = self.models.get(id).and_then(Weak::upgrade) {
            return lock;
        }
        self.models.retain(|_, w| w.strong_count() > 0);
        let lock = Arc::new(RwLock::new(()));
        self.models.insert(id.to_string(), Arc::downgrade(&lock));
        lock
    }

    pub(super) fn pair(&mut self, id: &str, variant: &str) -> (Arc<RwLock<()>>, Arc<Mutex<()>>) {
        let model = self.model(id);
        let key = (id.to_string(), variant.to_string());
        let var = if let Some(lock) = self.variants.get(&key).and_then(Weak::upgrade) {
            lock
        } else {
            self.variants.retain(|_, w| w.strong_count() > 0);
            let lock = Arc::new(Mutex::new(()));
            self.variants.insert(key, Arc::downgrade(&lock));
            lock
        };
        (model, var)
    }
}

/// The variant part of a derived weights file's name in `cache/weights/` (after `<model id>.<hash>.`).
fn cache_variant(format: WeightsFormat) -> String {
    match format {
        WeightsFormat::Onnx(s) => {
            let on: Vec<&str> = [("decode", s.decode), ("dynamic", s.dynamic), ("half", s.half)]
                .into_iter()
                .filter_map(|(n, on)| on.then_some(n))
                .collect();
            if on.is_empty() { "onnx".into() } else { format!("{}.onnx", on.join("-")) }
        }
        _ => "f16.folo".into(),
    }
}

impl Models {
    pub(super) fn weights_cache(&self) -> PathBuf {
        self.data_dir.join("cache").join("weights")
    }

    /// The derived weights of `m` for `GET /api/models/{id}/weights?format=f16|onnx`, opened (so removing an
    /// older copy never pulls it from under a request): `cache/weights/<model id>.<hash of the weights file's
    /// path, size and mtime>.<variant>` (`f16.folo`, or `onnx` with the switches, e.g. `decode-half.onnx`), written
    /// by the export code of `POST /api/models/{id}/export` the first time it is asked for (copies made from
    /// older versions of the model's file are then removed). `format` is [`WeightsFormat::F16`] or
    /// [`WeightsFormat::Onnx`].
    pub fn derived_weights(&self, m: &ModelRef, format: WeightsFormat) -> Result<(PathBuf, std::fs::File)> {
        let id = m.model.id.as_str();
        let variant = cache_variant(format);
        let dir = self.weights_cache();
        for attempt in 0..3 {
            let fk = FileKey::of(&m.path).map_err(|e| weights_io(e, &m.path, id))?;
            let mtime = fk
                .modified
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_nanos());
            let key = fnv1a(format!("{}|{}|{mtime}", fk.path.display(), fk.len).as_bytes());
            // Model ids have no dots, so `<id>.` never prefixes another model's files.
            let version = format!("{id}.{key:016x}.");
            let out = dir.join(format!("{version}{variant}"));
            let open = || match std::fs::File::open(&out) {
                Ok(f) => Ok(Some(f)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
                Err(e) => Err(Error::internal(format!("{}: {e}", out.display()))),
            };
            if let Some(f) = open()? {
                return Ok((out, f));
            }
            let (model_lock, variant_lock) =
                self.derived.lock().unwrap_or_else(|p| p.into_inner()).pair(id, &variant);
            let _model_guard = model_lock.read().unwrap_or_else(|p| p.into_inner());
            let _variant_guard = variant_lock.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(f) = open()? {
                return Ok((out, f));
            }
            // Its run may have been deleted (and its cache cleared, under the model's write lock) since the lookup.
            if !m.path.is_file() {
                return Err(Error::not_found("model", id));
            }
            // If the source file changed while we waited on the variant lock, retry against its new version.
            if attempt < 2 && FileKey::of(&m.path).ok().as_ref() != Some(&fk) {
                continue;
            }
            std::fs::create_dir_all(&dir)
                .map_err(|e| Error::internal(format!("creating {}: {e}", dir.display())))?;
            let file = engine::read_model_file(&m.path)
                .map_err(|e| if m.path.exists() { e } else { Error::not_found("model", id) })?;
            // Written to a temporary file and renamed into place, so a reader never sees a partial export.
            match format {
                WeightsFormat::Onnx(switches) => {
                    onnx_export(&file, switches)?.write_to(&out).map_err(engine_error)?
                }
                _ => export_folo_f16(&file, &out)?,
            }
            // Opened before releasing the locks; if a concurrent export for a newer file version pruned this
            // stale output between rename and open, retry against the newer version.
            let Some(f) = open()? else {
                if attempt < 2 {
                    continue;
                }
                return Err(Error::internal(format!("{} vanished", out.display())));
            };
            drop(_variant_guard);
            drop(_model_guard);
            // Prune stale exports of older versions only when any exist, and under `id`'s exclusive model write
            // lock after re-checking `FileKey`: while the write lock is held no other variant of `id` is writing
            // `.tmp` or publishing a newer version, so a late v1 export can never delete v2's files.
            if self.has_stale_weights(id, &version) {
                let _write_guard = model_lock.write().unwrap_or_else(|p| p.into_inner());
                if FileKey::of(&m.path).ok().as_ref() == Some(&fk) {
                    self.remove_weights(id, Some(&version));
                }
            }
            return Ok((out, f));
        }
        unreachable!()
    }

    /// Removes the cached derived weights of model `id` (when its run is deleted). Waits for any export of `id` in
    /// progress, so none is left behind, without blocking exports of other models.
    pub fn forget_weights(&self, id: &str) {
        let model_lock = self.derived.lock().unwrap_or_else(|p| p.into_inner()).model(id);
        let _guard = model_lock.write().unwrap_or_else(|p| p.into_inner());
        self.remove_weights(id, None);
    }

    /// True when the weights cache holds any file for model `id` that is not part of `keep_version`.
    pub(super) fn has_stale_weights(&self, id: &str, keep_version: &str) -> bool {
        let prefix = format!("{id}.");
        let Ok(entries) = std::fs::read_dir(self.weights_cache()) else { return false };
        entries.flatten().any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with(&prefix) && !name.starts_with(keep_version)
        })
    }

    /// Removes the cached derived weights of model `id` except those named `keep…` (the current version of its
    /// file). Always called under `id`'s exclusive model write lock, so no concurrent export of `id` is writing a
    /// `.tmp` file or publishing a newer version.
    pub(super) fn remove_weights(&self, id: &str, keep: Option<&str>) {
        let prefix = format!("{id}.");
        let Ok(entries) = std::fs::read_dir(self.weights_cache()) else { return };
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&prefix) && !keep.is_some_and(|k| name.starts_with(k)) {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
}
