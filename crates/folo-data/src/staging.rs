//! A dataset folder written whole or not at all: the files go to a staging folder beside it, `.<name>.partial`,
//! that is renamed to the folder once they are complete. A process killed part way leaves the staging folder, never
//! a folder that reads as a finished dataset. A marker file inside the staging folder, locked while its writer runs,
//! tells the next write to the same folder that what it finds is a stopped run's leftover (marker, no lock: removed
//! and written again), another process at work (refused) or not FOLO's (no marker: refused, never deleted).

use crate::error::{DataError, Result};
use crate::paths::shown;
use std::fs::{File, OpenOptions, TryLockError};
use std::io::{Error, ErrorKind, Write};
use std::path::{Path, PathBuf};

/// The marker file in a staging folder, locked by the process writing there.
const MARKER: &str = ".folo-partial";

/// A staging folder being written; see the module docs. Dropped before [`Staging::publish`] (an error or a panic
/// part way), it removes the staging folder.
#[derive(Debug)]
pub struct Staging {
    dir: PathBuf,
    path: PathBuf,
    marker: Option<File>,
}

impl Staging {
    /// Starts writing `dir`, which must not exist or be an empty folder by the time [`Staging::publish`] runs.
    /// A leftover staging folder of a stopped run is removed first. Fails with an [`ErrorKind::AlreadyExists`]
    /// [`DataError::Io`] when another process is writing `dir`, or when a folder FOLO did not make has the staging
    /// folder's name.
    pub fn begin(dir: &Path) -> Result<Staging> {
        let abs = std::path::absolute(dir).map_err(|e| DataError::io(dir, e))?;
        let (Some(parent), Some(name)) = (abs.parent(), abs.file_name()) else {
            return Err(DataError::Config(format!("{} does not name a folder to write", shown(dir))));
        };
        std::fs::create_dir_all(parent).map_err(|e| DataError::io(parent, e))?;
        let path = parent.join(format!(".{}.partial", name.to_string_lossy()));
        match std::fs::create_dir(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                clear_leftover(dir, &path)?;
                std::fs::create_dir(&path).map_err(|e| DataError::io(&path, e))?;
            }
            Err(e) => return Err(DataError::io(&path, e)),
        }
        let marker_path = path.join(MARKER);
        let mut marker = OpenOptions::new().write(true).create_new(true).open(&marker_path).map_err(|e| {
            // Without its marker the folder would be refused as not FOLO's from now on.
            let _ = std::fs::remove_dir(&path);
            DataError::io(&marker_path, e)
        })?;
        if marker.try_lock().is_err() {
            return Err(in_use(dir, &path));
        }
        let note = format!(
            "FOLO is writing {} here and renames this folder to it when complete. Left behind by a stopped \
             process, this folder is incomplete: the next write of that folder replaces it, or delete it.\n",
            shown(dir)
        );
        marker.write_all(note.as_bytes()).map_err(|e| DataError::io(&marker_path, e))?;
        Ok(Staging { dir: dir.to_path_buf(), path, marker: Some(marker) })
    }

    /// The staging folder to write into.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Moves the staging folder to the folder [`Staging::begin`] was given, in one rename (an empty folder there
    /// is replaced).
    pub fn publish(mut self) -> Result<()> {
        // Windows does not rename a folder with an open file in it: the marker goes first, then its handle.
        let marker = self.path.join(MARKER);
        std::fs::remove_file(&marker).map_err(|e| DataError::io(&marker, e))?;
        self.marker = None;
        let moved = match std::fs::rename(&self.path, &self.dir) {
            // Windows does not rename onto a folder, even an empty one; `remove_dir` removes only an empty one.
            Err(_) if self.dir.is_dir() => {
                std::fs::remove_dir(&self.dir).and_then(|()| std::fs::rename(&self.path, &self.dir))
            }
            r => r,
        };
        moved.map_err(|e| DataError::io(&self.dir, e))?;
        self.path = PathBuf::new();
        Ok(())
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        if !self.path.as_os_str().is_empty() {
            self.marker = None;
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}

/// Removes the staging folder `path` of `dir` when it is a stopped run's leftover: its marker is there and no
/// process holds it.
fn clear_leftover(dir: &Path, path: &Path) -> Result<()> {
    let marker = match OpenOptions::new().write(true).open(path.join(MARKER)) {
        Ok(f) => f,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            let m = format!(
                "{} is in the way: FOLO writes {} there first, and this folder is not FOLO's; move it away",
                shown(path),
                shown(dir)
            );
            return Err(DataError::io(path, Error::new(ErrorKind::AlreadyExists, m)));
        }
        Err(e) => return Err(DataError::io(path, e)),
    };
    match marker.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => return Err(in_use(dir, path)),
        Err(TryLockError::Error(e)) => return Err(DataError::io(path, e)),
    }
    drop(marker);
    std::fs::remove_dir_all(path).map_err(|e| DataError::io(path, e))
}

fn in_use(dir: &Path, path: &Path) -> DataError {
    let m = format!("another process is writing {} (in {}); wait for it to finish", shown(dir), shown(path));
    DataError::io(path, Error::new(ErrorKind::AlreadyExists, m))
}
