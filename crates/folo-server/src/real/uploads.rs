//! What finished uploads become: a registered dataset or a listed model.

use std::path::Path;

use axum::http::StatusCode;
use folo_api::{Dataset, Model, UploadKind, Uploaded};

use crate::error::{Error, Result};
use crate::models::file_model_id;
use crate::store::{WeightsFormat, weights_file_name};
use crate::uploads::Finished;
use crate::util::slug;

use super::RealStore;
use super::datasets::Naming;

impl RealStore {
    /// What the finished upload of `kind` with `fingerprint` became, while it still exists AND its on-disk content
    /// still matches what was adopted (bypassing the dataset recheck window so a modified worker copy is never
    /// silently reused).
    pub(super) fn uploaded(&self, kind: UploadKind, fingerprint: &str) -> Option<Uploaded> {
        let known = self.uploads.known(kind, fingerprint)?;
        match kind {
            UploadKind::Dataset => {
                let e = self.entry(&known.id).ok()?;
                let idx = self.fresh_index(&e).ok()?;
                let live = crate::index::Fingerprint::of(&idx.watched, &idx.excluded);
                if live != idx.fingerprint {
                    return None;
                }
                if let Some(expected) = known.content.as_deref()
                    && dataset_content_tag(&live) != expected
                {
                    return None;
                }
                Some(Uploaded { dataset: Some(Self::meta(&e.reg(), Some(&idx))), model: None })
            }
            UploadKind::Model => {
                let m = self.model(&known.id).ok()?;
                if let Some(expected) = known.content.as_deref()
                    && model_content_tag(&m.path).as_deref() != Some(expected)
                {
                    return None;
                }
                Some(Uploaded { dataset: None, model: Some(m.model) })
            }
        }
    }

    /// Moves an unpacked dataset upload into its own folder under `datasets/` and registers it, as `POST
    /// /api/datasets` would. Nothing stays behind when that fails.
    pub(super) fn adopt_dataset(&self, done: &Finished) -> Result<Dataset> {
        let dir = self.claim_dataset_dir(&slug(done.name(), "dataset"))?;
        let path = dir.to_string_lossy().into_owned();
        let registered = move_children(done.root(), &dir)
            .and_then(|()| self.register(Naming::Given(done.name().to_string()), path, done.format()))
            .map_err(|e| wrapped_archive(e, &dir));
        if registered.is_err() {
            let _ = std::fs::remove_dir_all(&dir);
        }
        registered
    }

    /// Moves an uploaded weights file to `models/<name>.folo` (the name reduced as for downloads, `-2`, ... when
    /// taken) once its header parses (`422 invalid_weights` otherwise), and returns its model.
    pub(super) fn adopt_model(&self, done: &Finished) -> Result<Model> {
        let file = done.file();
        crate::models::read_meta(&file).map_err(|e| {
            Error::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_weights",
                format!("the upload is not a readable .folo file: {}", e.body.message),
            )
            .hint("upload a FOLO weights file or checkpoint (docs/FORMAT.md)")
        })?;
        let dir = self.models.models_dir();
        std::fs::create_dir_all(&dir).map_err(|e| Error::io(format!("creating {}: {e}", dir.display())))?;
        let name = weights_file_name(done.name(), WeightsFormat::Stored);
        let stem = name.trim_end_matches(".folo");
        // Moved to the first free name in one step that fails when the name exists (folo_nn::io::publish_free):
        // two uploads never share or replace a file, and a server stopped meanwhile leaves no file under the name.
        let name = |n: u32| {
            let tag = if n == 1 { String::new() } else { format!("-{n}") };
            dir.join(format!("{stem}{tag}.folo"))
        };
        let path = folo_nn::io::publish_free(&file, name)
            .map_err(|e| Error::io(format!("moving {} to {}: {e}", file.display(), name(1).display())))?;
        self.model(&file_model_id(&path)).map(|m| m.model)
    }
}

/// The error of an uploaded archive that is not a dataset at its top but holds one in a folder (a zip of the dataset
/// folder): the folder it was unpacked into is removed right after, so the advice is to pack the dataset's contents
/// at the archive's top and upload again, not to register a folder that will be gone (x3-server-6).
fn wrapped_archive(e: Error, dir: &Path) -> Error {
    if e.body.code != "unknown_format" {
        return e;
    }
    let Some(below) = crate::index::dataset_below(dir, crate::index::DIAGNOSE_ENTRIES) else { return e };
    let inner = below.strip_prefix(dir).unwrap_or(&below).to_string_lossy().replace('\\', "/");
    Error::new(
        StatusCode::BAD_REQUEST,
        "unknown_format",
        format!("the archive is not a dataset at its top, but its folder '{inner}' holds one"),
    )
    .hint(format!(
        "pack what '{inner}' holds (data.yaml, images/ + labels/, or annotations/) at the archive's top level and \
         upload it again"
    ))
}

/// Stable content tag for an adopted dataset upload, matching the files/bytes/hash of its [`crate::index::Fingerprint`].
pub(super) fn dataset_content_tag(fp: &crate::index::Fingerprint) -> String {
    format!("{}:{}:{}", fp.files, fp.bytes, fp.hash)
}

/// Stable content tag for an adopted model upload (SHA-256 hex digest of the `.folo` file).
pub(super) fn model_content_tag(path: &Path) -> Option<String> {
    crate::label::dataset::sha256_file(path).ok()
}

/// Moves every entry of `from` into the folder `to` (renames: both are in the data dir).
fn move_children(from: &Path, to: &Path) -> Result<()> {
    let entries =
        std::fs::read_dir(from).map_err(|e| Error::io(format!("reading {}: {e}", from.display())))?;
    for e in entries {
        let e = e.map_err(|e| Error::io(format!("reading {}: {e}", from.display())))?;
        let dest = to.join(e.file_name());
        std::fs::rename(e.path(), &dest).map_err(|err| {
            Error::io(format!("moving {} to {}: {err}", e.path().display(), dest.display()))
        })?;
    }
    Ok(())
}
