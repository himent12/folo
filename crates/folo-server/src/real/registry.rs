//! The registry: `datasets.json`, the registered datasets (a plain list, as every version writes it, so an older
//! build reading the data dir sees them), and `datasets.removed.json`, the tombstones of unregistered ones, so a
//! dataset id is never handed to another dataset while runs still name it.

use std::path::{Path, PathBuf};

use folo_api::DatasetFormat;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::index::write_atomic;

use super::Registration;

/// The registry, both files.
#[derive(Debug, Default, Deserialize)]
pub(super) struct RegistryFile {
    pub datasets: Vec<Registration>,
    /// Unregistered datasets whose id runs still reference: registering the same dataset again (its canonical path
    /// and format) gives it that id back, and no other dataset gets it. Dropped at a registry write once no run the
    /// server has loaded references the id.
    #[serde(default)]
    pub tombstones: Vec<Tombstone>,
}

/// Which of the two files a write gives an id to (a registration, or a tombstone): that file is written first, so a
/// crash between the two writes leaves the id in both (a registration wins over its tombstone when read), never in
/// neither.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Gains {
    Registration,
    Tombstone,
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

pub(super) fn tombstones_path(dir: &Path) -> PathBuf {
    dir.join("datasets.removed.json")
}

/// The registry. A missing file is empty. `datasets.json` is a plain list of registrations; the object this
/// version's pre-release builds wrote (`{"datasets": [...], "tombstones": [...]}`) is read too, and written as a
/// list at the next write. An unreadable or malformed file is an error, so it is never overwritten (and its
/// contents lost) by the next write.
pub(super) fn read_registry(dir: &Path) -> Result<RegistryFile> {
    let path = registry_path(dir);
    let mut file = match read_text(&path)? {
        None => RegistryFile::default(),
        Some(text) if text.trim_start().starts_with('[') => {
            RegistryFile { datasets: parse(&path, &text)?, tombstones: Vec::new() }
        }
        Some(text) => parse(&path, &text)?,
    };
    let path = tombstones_path(dir);
    if let Some(text) = read_text(&path)? {
        for t in parse::<Vec<Tombstone>>(&path, &text)? {
            if !file.tombstones.iter().any(|x| x.id == t.id) {
                file.tombstones.push(t);
            }
        }
    }
    // A registration wins over a tombstone of its id (a crash between the two writes, see `Gains`).
    let RegistryFile { datasets, tombstones } = &mut file;
    tombstones.retain(|t| !datasets.iter().any(|r| r.id == t.id));
    Ok(file)
}

/// The text of the file at `path`; none when it does not exist.
fn read_text(path: &Path) -> Result<Option<String>> {
    match std::fs::read_to_string(path) {
        Ok(t) => Ok(Some(t)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::internal(format!("reading {}: {e}", path.display()))),
    }
}

fn parse<T: serde::de::DeserializeOwned>(path: &Path, text: &str) -> Result<T> {
    serde_json::from_str(text).map_err(|e| {
        Error::internal(format!("{} is malformed: {e}", path.display()))
            .hint("fix the JSON by hand or move the file away; FOLO will not overwrite it")
    })
}

/// Writes both files atomically, the one that `gains` an id first.
pub(super) fn write_registry(dir: &Path, file: &RegistryFile, gains: Gains) -> Result<()> {
    let datasets = (registry_path(dir), serde_json::to_vec_pretty(&file.datasets));
    let tombstones = (tombstones_path(dir), serde_json::to_vec_pretty(&file.tombstones));
    let order = match gains {
        Gains::Registration => [datasets, tombstones],
        Gains::Tombstone => [tombstones, datasets],
    };
    for (path, bytes) in order {
        let bytes = bytes.expect("registrations serialize");
        write_atomic(&path, &bytes)
            .map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))?;
    }
    Ok(())
}
