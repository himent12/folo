//! `datasets.json`: the registered datasets and the tombstones of unregistered ones, so a dataset id is never
//! handed to another dataset while runs still name it.

use std::path::{Path, PathBuf};

use folo_api::DatasetFormat;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::index::write_atomic;

use super::Registration;

/// The registry file.
#[derive(Debug, Default, Serialize, Deserialize)]
pub(super) struct RegistryFile {
    pub datasets: Vec<Registration>,
    /// Unregistered datasets whose id runs still reference: registering the same dataset again (its canonical path
    /// and format) gives it that id back, and no other dataset gets it. Dropped once no run references the id.
    #[serde(default)]
    pub tombstones: Vec<Tombstone>,
}

/// An unregistered dataset's identity and id.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Tombstone {
    pub id: String,
    /// The path as it was registered.
    pub path: String,
    /// Its canonical form when it was unregistered (absent when the folder was gone then).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub canonical: Option<PathBuf>,
    pub format: DatasetFormat,
    /// When it was unregistered.
    pub removed: String,
}

impl Tombstone {
    /// The tombstone of `reg`, unregistered now.
    pub fn of(reg: &Registration, removed: String) -> Tombstone {
        Tombstone {
            id: reg.id.clone(),
            path: reg.path.clone(),
            canonical: std::fs::canonicalize(&reg.path).ok(),
            format: reg.format,
            removed,
        }
    }

    /// The dataset at canonical path `canonical`, read as `format`, is this one.
    pub fn is(&self, canonical: &Path, format: DatasetFormat) -> bool {
        self.format == format
            && (self.canonical.as_deref() == Some(canonical)
                || std::fs::canonicalize(&self.path).is_ok_and(|p| p == canonical))
    }
}

impl RegistryFile {
    /// Drops the tombstones whose id no run references any more.
    pub fn prune(&mut self, referenced: &dyn Fn(&str) -> bool) {
        self.tombstones.retain(|t| referenced(&t.id));
    }

    /// `id` belongs to a registration or a tombstone.
    pub fn holds(&self, id: &str) -> bool {
        self.datasets.iter().any(|r| r.id == id) || self.tombstones.iter().any(|t| t.id == id)
    }
}

pub(super) fn registry_path(dir: &Path) -> PathBuf {
    dir.join("datasets.json")
}

/// The registry. A missing file is an empty registry, and a plain list of registrations (as versions before
/// tombstones wrote it) has none; an unreadable or malformed file is an error, so it is never overwritten (and its
/// registrations lost) by the next write.
pub(super) fn read_registry(dir: &Path) -> Result<RegistryFile> {
    let path = registry_path(dir);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(RegistryFile::default()),
        Err(e) => return Err(Error::internal(format!("reading {}: {e}", path.display()))),
    };
    let malformed = |e: serde_json::Error| {
        Error::internal(format!("{} is malformed: {e}", path.display()))
            .hint("fix the JSON by hand or move the file away; FOLO will not overwrite it")
    };
    if text.trim_start().starts_with('[') {
        let datasets = serde_json::from_str(&text).map_err(malformed)?;
        return Ok(RegistryFile { datasets, tombstones: Vec::new() });
    }
    serde_json::from_str(&text).map_err(malformed)
}

pub(super) fn write_registry(dir: &Path, file: &RegistryFile) -> Result<()> {
    let path = registry_path(dir);
    let bytes = serde_json::to_vec_pretty(file).expect("registrations serialize");
    write_atomic(&path, &bytes).map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))
}
