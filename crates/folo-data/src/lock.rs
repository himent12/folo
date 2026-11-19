//! An exclusive OS lock on a file, taken without waiting. `folo serve`'s `serve.lock`, a cloud run's `cloud.lock`
//! and the marker of a dataset being written ([`crate::unfinished`]) use it. The OS releases the lock when the file is
//! closed or its process ends, so a lock that cannot be taken is a live process's.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

/// Opens `path` for reading and writing (creating it when missing, never truncating it) and locks it exclusively:
/// `None` when another handle, in this process or another one, holds the lock. The lock goes with the file.
pub fn try_lock(path: &Path) -> std::io::Result<Option<File>> {
    let file = OpenOptions::new().create(true).truncate(false).read(true).write(true).open(path)?;
    match file.try_lock() {
        Ok(()) => Ok(Some(file)),
        Err(TryLockError::WouldBlock) => Ok(None),
        Err(TryLockError::Error(e)) => Err(e),
    }
}
