//! A dataset folder being written, recognisable as unfinished until it is complete. `synthetic::write_yolo` and
//! `convert::convert` write in place, in the folder they are given, with a marker file ([`MARKER`]) in it from before
//! their first file to after their last. A process stopped part way leaves the marker: every reader refuses the
//! folder ([`DataError::Unfinished`]) and the next write of the folder clears it and writes it whole.
//!
//! The marker is also the writers' mutex: a writer opens or creates it and locks it ([`crate::lock::try_lock`]), and
//! one that cannot take the lock stops before touching anything. The holder writes [`NOTE`] into the marker before
//! anything else and empties it before removing it at the end, so once a writer holds the marker:
//! - a marker holding the note is a stopped writer's, and the folder's other entries are its partial output;
//! - an empty marker was never written under. Beside other entries (a dataset finished as this writer started, or
//!   its writer stopped right before removing the marker) it is removed and the folder refused as not empty.
//!
//! A folder with other entries and no marker is refused untouched, as is a marker that is not a plain file holding
//! the note or nothing. Nothing outside the folder is touched.

use crate::error::{DataError, Result};
use crate::paths::shown;
use std::fs::File;
use std::io::{Error, ErrorKind, Read, Seek, Write};
use std::path::{Path, PathBuf};

/// The marker file in a dataset folder being written.
pub const MARKER: &str = ".folo-unfinished";

/// What the writer holding the marker keeps in it.
const NOTE: &str = "FOLO is writing a dataset in this folder. If no FOLO command is writing it, the command was \
                    stopped part way: run it again to write the folder whole, or delete the folder.\n";

/// A dataset folder this process writes; see the module docs. Dropped without [`Writing::finish`] (an error part
/// way), the folder stays marked unfinished.
#[derive(Debug)]
pub struct Writing {
    marker: File,
    path: PathBuf,
}

impl Writing {
    /// Takes `dir` for writing, creating it when missing. `dir` must be empty or hold a stopped write, whose files
    /// are removed first. Fails with an [`ErrorKind::AlreadyExists`] [`DataError::Io`] when `dir` holds anything
    /// else or another process is writing it.
    pub fn begin(dir: &Path) -> Result<Writing> {
        std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
        let path = dir.join(MARKER);
        let marked = std::fs::symlink_metadata(&path);
        if marked.as_ref().is_ok_and(|m| !m.is_file()) || (marked.is_err() && others(dir)?) {
            return Err(not_empty(dir));
        }
        let Some(mut marker) = crate::lock::try_lock(&path).map_err(|e| DataError::io(&path, e))? else {
            let m = format!("another FOLO command is writing {}; wait for it to finish", shown(dir));
            return Err(DataError::io(dir, Error::new(ErrorKind::AlreadyExists, m)));
        };
        let mut head = Vec::new();
        (&marker).take(NOTE.len() as u64 + 1).read_to_end(&mut head).map_err(|e| DataError::io(&path, e))?;
        if head == NOTE.as_bytes() {
            clear(dir)?;
        } else if !head.is_empty() {
            return Err(not_empty(dir));
        } else if others(dir)? {
            let _ = std::fs::remove_file(&path);
            return Err(not_empty(dir));
        } else {
            marker.write_all(NOTE.as_bytes()).map_err(|e| DataError::io(&path, e))?;
        }
        Ok(Writing { marker, path })
    }

    /// Marks the folder complete: removes the marker.
    pub fn finish(mut self) -> Result<()> {
        // Emptied while still locked: a process that opened the marker before it goes finds it empty beside a
        // complete dataset, which it never clears.
        match self.marker.set_len(0).and_then(|()| std::fs::remove_file(&self.path)) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
            Err(e) => {
                // A stopped write again, so the same command run again writes the folder whole.
                let _ = self.marker.rewind().and_then(|()| self.marker.write_all(NOTE.as_bytes()));
                let m = format!(
                    "could not remove {} ({e}): the folder stays marked unfinished; run the same command again",
                    shown(&self.path)
                );
                Err(DataError::io(&self.path, Error::new(e.kind(), m)))
            }
        }
    }
}

/// Fails with [`DataError::Unfinished`] when `dir` holds the marker: a dataset being written, or a stopped write.
pub(crate) fn check(dir: &Path) -> Result<()> {
    match std::fs::symlink_metadata(dir.join(MARKER)) {
        Ok(_) => Err(DataError::Unfinished(dir.to_path_buf())),
        Err(_) => Ok(()),
    }
}

/// Whether `dir` has an entry other than the marker.
fn others(dir: &Path) -> Result<bool> {
    let mut entries = std::fs::read_dir(dir).map_err(|e| DataError::io(dir, e))?;
    Ok(entries.any(|e| e.map_or(true, |e| e.file_name() != MARKER)))
}

/// Removes every entry of `dir` but the marker. A link is removed as a link, never what it points to:
/// `remove_dir_all` does not follow links (in `dir` or below), a link reports `is_dir() == false`, and one that
/// `remove_file` does not take (a Windows junction or folder symlink) goes with `remove_dir`, which removes the link.
fn clear(dir: &Path) -> Result<()> {
    for e in std::fs::read_dir(dir).map_err(|e| DataError::io(dir, e))? {
        let e = e.map_err(|e| DataError::io(dir, e))?;
        if e.file_name() == MARKER {
            continue;
        }
        let p = e.path();
        let removed = match e.file_type() {
            Ok(t) if t.is_dir() => std::fs::remove_dir_all(&p),
            _ => std::fs::remove_file(&p).or_else(|err| std::fs::remove_dir(&p).map_err(|_| err)),
        };
        removed.map_err(|err| DataError::io(&p, err))?;
    }
    Ok(())
}

fn not_empty(dir: &Path) -> DataError {
    let m = format!("{} exists and is not an empty folder", shown(dir));
    DataError::io(dir, Error::new(ErrorKind::AlreadyExists, m))
}
