//! A dataset folder being written, recognisable as unfinished until it is complete (issue #316).
//! Every writer (`synthetic::write_yolo`, `convert::convert`, the server's sample sets) starts with [`claim`]: it
//! creates a [`MARKER`] file in an empty output folder, and [`Claim::finish`] removes it as the last step; a process
//! stopped part way leaves it. Readers refuse a folder holding it ([`DataError::Unfinished`]). FOLO never reads the
//! marker's content and never clears or deletes a marked folder.

use crate::error::{DataError, Result};
use crate::paths::shown;
use std::io::{Error, ErrorKind, Write};
use std::path::{Path, PathBuf};

/// The marker file in a dataset folder being written.
pub const MARKER: &str = ".folo-unfinished";

/// What to do about a marked folder (the CLI's and the server's hint).
pub const HINT: &str = "it holds an unfinished dataset: if no folo command is writing it, delete the folder and run the command again";

/// For a person who opens the marker.
const NOTE: &str = "FOLO is writing this dataset, or was stopped while writing it.\nIf no `folo` command is running, \
                    delete this folder and run the command again.\n";

/// An output folder this call owns as its writer ([`claim`]): it created the marker in it.
#[must_use = "the folder stays marked unfinished until `finish`"]
#[derive(Debug)]
pub struct Claim(PathBuf);

impl Claim {
    /// The claimed folder.
    pub fn dir(&self) -> &Path {
        &self.0
    }

    /// Marks the folder complete: removes the marker, the writer's last step. On Windows another program (a virus
    /// scanner, the search indexer) may hold the new file open for a moment without sharing deletion, so a removal
    /// refused that way is tried again for about half a second.
    pub fn finish(self) -> Result<()> {
        let marker = self.0.join(MARKER);
        let mut tries = 0;
        let removed = loop {
            match std::fs::remove_file(&marker) {
                Err(e) if tries < 10 && is_held(&e) => {
                    tries += 1;
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
                done => break done,
            }
        };
        removed.map_err(|e| {
            let m = format!(
                "the dataset is complete, but {} could not be removed ({e}): delete that file",
                shown(&marker)
            );
            DataError::io(&marker, Error::new(e.kind(), m))
        })
    }
}

/// Claims `dir` for one writer: creates it when missing, then the marker in it (`create_new`, so only one writer
/// can), then, holding the marker, checks that nothing else is in it. Fails with a [`DataError::Io`] on `dir` saying
/// it "exists and is not an empty folder", of kind [`ErrorKind::AlreadyExists`] when the marker was there (another
/// writer's folder, running or stopped) and [`ErrorKind::DirectoryNotEmpty`] when other files are (or `dir` is a
/// file); either way it leaves the folder as it was.
pub fn claim(dir: &Path) -> Result<Claim> {
    let in_use = |kind| {
        DataError::io(dir, Error::new(kind, format!("{} exists and is not an empty folder", shown(dir))))
    };
    std::fs::create_dir_all(dir).map_err(|e| match e.kind() {
        ErrorKind::AlreadyExists => in_use(ErrorKind::DirectoryNotEmpty),
        _ => DataError::io(dir, e),
    })?;
    let marker = dir.join(MARKER);
    let mut file = std::fs::File::create_new(&marker).map_err(|e| match e.kind() {
        ErrorKind::AlreadyExists => in_use(ErrorKind::AlreadyExists),
        _ => DataError::io(&marker, e),
    })?;
    let written = file.write_all(NOTE.as_bytes()).map_err(|e| DataError::io(&marker, e));
    drop(file);
    let others = std::fs::read_dir(dir)
        .map_err(|e| DataError::io(dir, e))
        .map(|mut rd| rd.any(|e| e.map_or(true, |e| e.file_name() != MARKER)));
    match (written, others) {
        (Ok(()), Ok(false)) => Ok(Claim(dir.to_path_buf())),
        (w, o) => {
            let _ = std::fs::remove_file(&marker);
            Err(w.and(o).err().unwrap_or_else(|| in_use(ErrorKind::DirectoryNotEmpty)))
        }
    }
}

/// Whether a failed removal can be another program holding the file open for a moment (Windows: a sharing
/// violation, error 32, or access denied while the file is pending deletion).
fn is_held(e: &Error) -> bool {
    e.kind() == ErrorKind::PermissionDenied || (cfg!(windows) && e.raw_os_error() == Some(32))
}

/// Fails with [`DataError::Unfinished`] when `dir` holds the marker.
pub(crate) fn check(dir: &Path) -> Result<()> {
    match std::fs::symlink_metadata(dir.join(MARKER)) {
        Ok(_) => Err(DataError::Unfinished(dir.to_path_buf())),
        Err(_) => Ok(()),
    }
}
