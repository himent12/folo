//! The cloud module's files: owner-only atomic writes (credentials and worker tokens live in them), the data dir's
//! cloud lock and the lock of the one process that follows a cloud run.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Writes `bytes` to `path` atomically (a temporary file next to it, then a rename), readable by the owner only.
/// On Unix the file is created with mode 0600. On Windows a new file inherits the access rules of its folder: the
/// data dir sits in the user's profile (`~/.folo`), which only the user, administrators and SYSTEM can read.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let written = options.open(&tmp).and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()));
    // Briefly: these are written under `cloud/.lock` or the run's state, which a long wait would stall.
    let brief = folo_nn::io::RenameRetry::BRIEF;
    written.and_then(|()| folo_nn::io::rename_over_with(&tmp, path, brief).map(drop)).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// Waits for the exclusive OS lock on `path` (created when missing) and holds it until the returned file is
/// dropped. The data dir's `cloud/.lock` is held across every read-modify-write of `connections.json` and of a
/// ledger record, so two processes on one data dir (a server and `folo cloud`) never interleave theirs. The lock
/// belongs to the open file, so two threads of one process exclude each other too.
pub(crate) fn lock(path: &Path) -> std::io::Result<File> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(path)?;
    file.lock()?;
    Ok(file)
}

/// `runs/<id>/cloud.lock`, held by the one process that follows a cloud run, from the moment it registers or picks
/// up the run until the run is released (or the process lets go of it at shutdown). The file stays; the lock is what
/// counts.
pub(crate) struct RunLock {
    /// Locked; dropping it unlocks.
    _file: File,
}

/// The lock file of the run in `run_dir`.
pub(crate) fn run_lock_path(run_dir: &Path) -> PathBuf {
    run_dir.join("cloud.lock")
}

/// Takes `run_dir`'s [`RunLock`] if no process holds it: `None` when one does. A run folder that is gone is a
/// `NotFound` error (no process can follow a deleted run).
pub(crate) fn try_lock_run(run_dir: &Path) -> std::io::Result<Option<RunLock>> {
    let file = OpenOptions::new().create(true).truncate(false).write(true).open(run_lock_path(run_dir))?;
    match file.try_lock() {
        Ok(()) => Ok(Some(RunLock { _file: file })),
        Err(std::fs::TryLockError::WouldBlock) => Ok(None),
        Err(std::fs::TryLockError::Error(e)) => Err(e),
    }
}

/// `n` random bytes as lowercase hex (worker tokens, instance name nonces).
pub(crate) fn random_hex(n: usize) -> std::io::Result<String> {
    let mut bytes = vec![0u8; n];
    getrandom::fill(&mut bytes).map_err(|e| std::io::Error::other(format!("no random bytes: {e}")))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    #[test]
    fn private_files_replace_whole_and_run_locks_exclude() {
        let dir = TestDir::new("cloud-files");
        let p = dir.join("cloud").join("connections.json");
        write_private(&p, b"one").unwrap();
        write_private(&p, b"two").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"two");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o777, 0o600);
        }
        let leftovers = std::fs::read_dir(p.parent().unwrap()).unwrap().count();
        assert_eq!(leftovers, 1, "no temporary files stay");

        let run = dir.join("run_1");
        assert_eq!(
            try_lock_run(&run).err().map(|e| e.kind()),
            Some(std::io::ErrorKind::NotFound),
            "no folder, no lock"
        );
        std::fs::create_dir_all(&run).unwrap();
        let held = try_lock_run(&run).unwrap().expect("free");
        assert!(try_lock_run(&run).unwrap().is_none(), "a second handle is refused, also in one process");
        drop(held);
        assert!(try_lock_run(&run).unwrap().is_some());
        let hex = random_hex(32).unwrap();
        assert_eq!(hex.len(), 64);
        assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()) && hex != random_hex(32).unwrap());
    }
}
