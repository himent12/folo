//! The registry: `datasets.json`, the registered datasets (a plain list, as every version writes it, so an older
//! build reading the data dir sees them), and `datasets.removed.json`, the tombstones of unregistered ones, so a
//! dataset id is never handed to another dataset while runs still name it.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use folo_api::DatasetFormat;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::index::write_atomic;

use super::Registration;

/// The registry, both files.
#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct RegistryFile {
    pub datasets: Vec<Registration>,
    /// Unregistered datasets whose id runs still reference: registering the same dataset again (its canonical path
    /// and format) gives it that id back, and no other dataset gets it. Dropped at a registry write once no run the
    /// server has loaded references the id.
    #[serde(default)]
    pub tombstones: Vec<Tombstone>,
    /// Read from the object form, whose tombstones are inside `datasets.json` (see [`write_registry`]).
    #[serde(skip)]
    pub object: bool,
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
            RegistryFile { datasets: parse(&path, &text)?, ..RegistryFile::default() }
        }
        Some(text) => RegistryFile { object: true, ..parse(&path, &text)? },
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
    let RegistryFile { datasets, tombstones, .. } = &mut file;
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

/// Writes both files atomically, the one that `gains` an id first. A registry read from the object form has the
/// only copy of its tombstones in the `datasets.json` the list replaces, so the tombstones are written first then,
/// whatever the write gains: until the list replaces it, the object still holds every id it held.
pub(super) fn write_registry(dir: &Path, file: &RegistryFile, gains: Gains) -> Result<()> {
    write_with(dir, file, gains, &mut write_atomic)
}

/// [`write_registry`], each file written by `write` (tests make one fail).
pub(super) fn write_with(
    dir: &Path,
    file: &RegistryFile,
    gains: Gains,
    write: &mut dyn FnMut(&Path, &[u8]) -> std::io::Result<()>,
) -> Result<()> {
    let datasets = (registry_path(dir), serde_json::to_vec_pretty(&file.datasets));
    let tombstones = (tombstones_path(dir), serde_json::to_vec_pretty(&file.tombstones));
    let order = match gains {
        Gains::Registration if !file.object => [datasets, tombstones],
        _ => [tombstones, datasets],
    };
    for (path, bytes) in order {
        let bytes = bytes.expect("registrations serialize");
        write(&path, &bytes).map_err(|e| Error::internal(format!("writing {}: {e}", path.display())))?;
    }
    Ok(())
}

/// Stat fingerprint of one registry file on disk (`None` when the file does not exist).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct FileStamp {
    len: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    dev: u64,
    #[cfg(unix)]
    ino: u64,
    #[cfg(unix)]
    ctime: (i64, i64),
    #[cfg(windows)]
    created: u64,
}

pub(super) type RegistryStamps = (Option<FileStamp>, Option<FileStamp>);

fn stamp_file(path: &Path) -> std::io::Result<Option<FileStamp>> {
    match std::fs::metadata(path) {
        Ok(md) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                Ok(Some(FileStamp {
                    len: md.len(),
                    modified: md.modified().ok(),
                    dev: md.dev(),
                    ino: md.ino(),
                    ctime: (md.ctime(), md.ctime_nsec()),
                }))
            }
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                Ok(Some(FileStamp {
                    len: md.len(),
                    modified: md.modified().ok(),
                    created: md.creation_time(),
                }))
            }
            #[cfg(not(any(unix, windows)))]
            {
                Ok(Some(FileStamp { len: md.len(), modified: md.modified().ok() }))
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

fn registry_stamps(dir: &Path) -> std::io::Result<RegistryStamps> {
    Ok((stamp_file(&registry_path(dir))?, stamp_file(&tombstones_path(dir))?))
}

struct CachedRegistry {
    stamps: RegistryStamps,
    file: RegistryFile,
}

/// Stat-validated in-memory cache of `datasets.json` and `datasets.removed.json`: avoids redundant disk reads and
/// JSON deserialization on repeated registry operations while picking up external modifications immediately.
#[derive(Default)]
pub(super) struct RegistryCache {
    cached: Option<CachedRegistry>,
    #[cfg(test)]
    pub(super) disk_reads: u64,
}

impl RegistryCache {
    /// Seeds the cache with a [`RegistryFile`] just read from `dir` when its on-disk stamps remained stable.
    pub(super) fn seed(&mut self, dir: &Path, before: Option<RegistryStamps>, file: &RegistryFile) {
        if let (Some(before), Ok(after)) = (before, registry_stamps(dir))
            && before == after
        {
            self.cached = Some(CachedRegistry { stamps: after, file: file.clone() });
        }
    }

    /// Reads the registry, serving from memory when both files' on-disk stamps are unchanged. Never caches errors.
    pub(super) fn read(&mut self, dir: &Path) -> Result<RegistryFile> {
        let before = registry_stamps(dir).ok();
        if let (Some(stamps), Some(cached)) = (&before, &self.cached)
            && *stamps == cached.stamps
        {
            return Ok(cached.file.clone());
        }
        #[cfg(test)]
        {
            self.disk_reads += 1;
        }
        match read_registry(dir) {
            Ok(file) => {
                // Only cache if neither file changed while we were reading it.
                self.cached = None;
                self.seed(dir, before, &file);
                Ok(file)
            }
            Err(e) => {
                self.cached = None;
                Err(e)
            }
        }
    }

    /// Writes `file` atomically and refreshes the cache. Invalidates before writing so a partial/failed write
    /// never leaves stale cached state.
    pub(super) fn write(&mut self, dir: &Path, file: &mut RegistryFile, gains: Gains) -> Result<()> {
        self.cached = None;
        write_registry(dir, file, gains)?;
        // `write_registry` always emits the plain-list `datasets.json` and separate `datasets.removed.json`.
        file.object = false;
        if let Ok(stamps) = registry_stamps(dir) {
            self.cached = Some(CachedRegistry { stamps, file: file.clone() });
        }
        Ok(())
    }

    /// Current stamps of both registry files before an initial read at startup.
    pub(super) fn stamps(dir: &Path) -> Option<RegistryStamps> {
        registry_stamps(dir).ok()
    }
}
