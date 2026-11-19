//! A dataset folder being written, recognisable as unfinished until it is complete (issue #316).
//! `synthetic::write_yolo` and `convert::convert` create a [`MARKER`] file in their output folder before their first
//! file and remove it as their last step; a process stopped part way leaves it. Readers refuse a folder holding it
//! ([`DataError::Unfinished`]). FOLO never reads the marker's content and never clears or deletes a marked folder.

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

/// Creates `dir` when missing, then the marker in it. When the marker exists (another writer's folder, running or
/// stopped) fails with an [`ErrorKind::AlreadyExists`] [`DataError::Io`] on the marker and changes nothing.
pub(crate) fn begin(dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).map_err(|e| DataError::io(dir, e))?;
    let marker = dir.join(MARKER);
    let mut file = std::fs::File::create_new(&marker).map_err(|e| match e.kind() {
        ErrorKind::AlreadyExists => {
            let m = format!("{} exists and is not an empty folder", shown(dir));
            DataError::io(&marker, Error::new(ErrorKind::AlreadyExists, m))
        }
        _ => DataError::io(&marker, e),
    })?;
    file.write_all(NOTE.as_bytes()).map_err(|e| DataError::io(&marker, e))?;
    Ok(marker)
}

/// Marks the folder complete: removes `marker`, the writer's last step.
pub(crate) fn finish(marker: &Path) -> Result<()> {
    std::fs::remove_file(marker).map_err(|e| {
        let m = format!(
            "the dataset is complete, but {} could not be removed ({e}): delete that file",
            shown(marker)
        );
        DataError::io(marker, Error::new(e.kind(), m))
    })
}

/// Fails with [`DataError::Unfinished`] when `dir` holds the marker.
pub(crate) fn check(dir: &Path) -> Result<()> {
    match std::fs::symlink_metadata(dir.join(MARKER)) {
        Ok(_) => Err(DataError::Unfinished(dir.to_path_buf())),
        Err(_) => Ok(()),
    }
}
