//! What finished uploads become: a registered dataset or a listed model.

use std::path::Path;

use axum::http::StatusCode;
use folo_api::{Dataset, Model, UploadKind, Uploaded};

use crate::error::{Error, Result};
use crate::models::file_model_id;
use crate::store::{Store, WeightsFormat, weights_file_name};
use crate::uploads::Finished;
use crate::util::slug;

use super::RealStore;
use super::datasets::Naming;

impl RealStore {
    /// What the finished upload of `kind` with `fingerprint` became, while it still exists (the dataset registered,
    /// the model listed).
    pub(super) fn uploaded(&self, kind: UploadKind, fingerprint: &str) -> Option<Uploaded> {
        let id = self.uploads.known(kind, fingerprint)?;
        match kind {
            UploadKind::Dataset => {
                self.get_dataset(&id).ok().map(|d| Uploaded { dataset: Some(d), model: None })
            }
            UploadKind::Model => {
                self.model(&id).ok().map(|m| Uploaded { dataset: None, model: Some(m.model) })
            }
        }
    }

    /// Moves an unpacked dataset upload into its own folder under `datasets/` and registers it, as `POST
    /// /api/datasets` would. Nothing stays behind when that fails.
    pub(super) fn adopt_dataset(&self, done: &Finished) -> Result<Dataset> {
        let dir = self.claim_dataset_dir(&slug(done.name(), "dataset"))?;
        let path = dir.to_string_lossy().into_owned();
        let registered = move_children(done.root(), &dir)
            .and_then(|()| self.register(Naming::Given(done.name().to_string()), path, done.format()));
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
        // create_new claims the name, so two uploads never share a file; the upload then replaces it atomically.
        let mut n = 1;
        let path = loop {
            let tag = if n == 1 { String::new() } else { format!("-{n}") };
            let p = dir.join(format!("{stem}{tag}.folo"));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
                Ok(_) => break p,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => n += 1,
                Err(e) => return Err(Error::io(format!("creating {}: {e}", p.display()))),
            }
        };
        if let Err(e) = std::fs::rename(&file, &path) {
            let _ = std::fs::remove_file(&path);
            return Err(Error::io(format!("moving {} to {}: {e}", file.display(), path.display())));
        }
        self.model(&file_model_id(&path)).map(|m| m.model)
    }
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
