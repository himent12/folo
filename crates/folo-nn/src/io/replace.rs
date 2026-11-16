//! Replacing a file by renaming a finished temporary file over it ([`rename_over`]), with a bounded retry while
//! another program holds the target open.
//!
//! On Windows a rename over a file fails with `ERROR_ACCESS_DENIED` (5), `ERROR_SHARING_VIOLATION` (32) or
//! `ERROR_LOCK_VIOLATION` (33) for as long as any process has the target open without `FILE_SHARE_DELETE`: Python's
//! `open()`, an antivirus or backup scan, an editor. Such readers usually let go within milliseconds, so the rename
//! is tried again with a growing wait. On other platforms a rename over an open file succeeds, and nothing is
//! retried. The success path is a single `std::fs::rename`, as before: same atomicity, no extra sync.

use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

/// How [`rename_over_with`] waits for a target that another program holds open: `first`, doubled after every
/// attempt up to `cap`, until `budget` has passed since the first attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenameRetry {
    /// The wait after the first failed attempt.
    pub first: Duration,
    /// The longest single wait.
    pub cap: Duration,
    /// The total time after which the rename gives up.
    pub budget: Duration,
}

impl RenameRetry {
    /// 10 ms, doubling, at most 1 s per wait and 10 s in all. Readers seen in practice (a script that opens the
    /// checkpoint for a few milliseconds, a virus scan) let go well within that; one that holds the file longer is
    /// a real lock, and the trainer's epoch loop must then fail with an error that says so rather than hang
    /// silently.
    pub const DEFAULT: RenameRetry = RenameRetry {
        first: Duration::from_millis(10),
        cap: Duration::from_secs(1),
        budget: Duration::from_secs(10),
    };
}

/// Renames `tmp` over `path` ([`rename_over_with`] with [`RenameRetry::DEFAULT`]). The caller removes `tmp` when
/// this fails.
pub fn rename_over(tmp: &Path, path: &Path) -> io::Result<()> {
    rename_over_with(tmp, path, RenameRetry::DEFAULT).map(drop)
}

/// Renames `tmp` over `path`, retrying while another program holds `path` open (Windows only, see the module
/// docs) for at most `retry.budget`, and returns how long it waited (zero when the first attempt succeeded). Any
/// other error is returned at once. The error after the budget names `path` and says another program holds it
/// open. `tmp` is left in place either way: the caller removes it when this fails.
pub fn rename_over_with(tmp: &Path, path: &Path, retry: RenameRetry) -> io::Result<Duration> {
    let start = Instant::now();
    let mut wait = retry.first;
    let mut waited = Duration::ZERO;
    loop {
        let err = match std::fs::rename(tmp, path) {
            Ok(()) => return Ok(waited),
            Err(e) if held_open(&e, path) => e,
            Err(e) => return Err(e),
        };
        let spent = start.elapsed();
        if spent >= retry.budget {
            return Err(io::Error::new(
                err.kind(),
                format!(
                    "another program holds {} open, so it cannot be replaced (retried for {:.1} s; this is not a \
                     permissions problem: close the program that has the file open): {err}",
                    path.display(),
                    spent.as_secs_f64()
                ),
            ));
        }
        let nap = wait.min(retry.budget - spent);
        std::thread::sleep(nap);
        waited += nap;
        wait = (wait * 2).min(retry.cap);
    }
}

/// Whether `e`, from renaming over `path`, means another program holds `path` open (and may let go). Access denied
/// is also what a directory or a read-only file at `path` gives: those fail at once.
#[cfg(windows)]
fn held_open(e: &io::Error, path: &Path) -> bool {
    const ACCESS_DENIED: i32 = 5;
    const SHARING_VIOLATION: i32 = 32;
    const LOCK_VIOLATION: i32 = 33;
    match e.raw_os_error() {
        Some(SHARING_VIOLATION | LOCK_VIOLATION) => true,
        Some(ACCESS_DENIED) => {
            std::fs::metadata(path).is_ok_and(|m| m.is_file() && !m.permissions().readonly())
        }
        _ => false,
    }
}

#[cfg(not(windows))]
fn held_open(_: &io::Error, _: &Path) -> bool {
    false
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::{Duration, Instant};

    use folo_test_support::TestDir;

    use super::RenameRetry;
    use crate::io::safetensors::{write, write_file, write_file_with};

    fn meta(v: &str) -> BTreeMap<String, String> {
        BTreeMap::from([("v".to_string(), v.to_string())])
    }

    /// The bytes [`write_file`] writes for `meta(v)` and no tensors.
    fn bytes(v: &str) -> Vec<u8> {
        let mut out = Vec::new();
        write(&mut out, Some(&meta(v)), &[]).unwrap();
        out
    }

    /// The names in `dir`, sorted (a temporary file left behind shows up here).
    fn names(dir: &TestDir) -> Vec<String> {
        let mut n: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        n.sort();
        n
    }

    const SHORT: RenameRetry = RenameRetry {
        first: Duration::from_millis(10),
        cap: Duration::from_millis(50),
        budget: Duration::from_millis(300),
    };

    #[test]
    fn a_plain_write_replaces_the_file() {
        let dir = TestDir::new("nn-replace-plain");
        let path = dir.join("last.folo");
        write_file(&path, Some(&meta("old")), &[]).unwrap();
        write_file(&path, Some(&meta("new")), &[]).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes("new"));
        assert_eq!(names(&dir), ["last.folo"]);
    }

    #[test]
    fn an_error_that_is_not_a_held_file_fails_at_once() {
        let dir = TestDir::new("nn-replace-dir");
        let path = dir.join("last.folo");
        std::fs::create_dir(&path).unwrap();
        let slow = RenameRetry { budget: Duration::from_secs(5), ..SHORT };
        let t = Instant::now();
        assert!(write_file_with(&path, Some(&meta("new")), &[], slow).is_err());
        assert!(t.elapsed() < Duration::from_secs(1), "waited {:?} for a permanent error", t.elapsed());
        assert_eq!(names(&dir), ["last.folo"]);
    }

    /// `path` opened for reading the way Python's `open()` does: sharing read and write, not delete.
    #[cfg(windows)]
    fn hold(path: &std::path::Path) -> std::fs::File {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 1;
        const FILE_SHARE_WRITE: u32 = 2;
        std::fs::OpenOptions::new()
            .read(true)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE)
            .open(path)
            .unwrap()
    }

    #[cfg(windows)]
    #[test]
    fn a_write_waits_for_a_reader_that_lets_go() {
        let dir = TestDir::new("nn-replace-reader");
        let path = dir.join("last.folo");
        write_file(&path, Some(&meta("old")), &[]).unwrap();
        let held = hold(&path);
        let reader = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(held);
        });
        let r = write_file(&path, Some(&meta("new")), &[]);
        reader.join().unwrap();
        r.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), bytes("new"));
        assert_eq!(names(&dir), ["last.folo"]);
    }

    #[cfg(windows)]
    #[test]
    fn a_reader_that_holds_on_past_the_budget_fails_the_write() {
        let dir = TestDir::new("nn-replace-lock");
        let path = dir.join("last.folo");
        write_file(&path, Some(&meta("old")), &[]).unwrap();
        let held = hold(&path);
        let t = Instant::now();
        let err = write_file_with(&path, Some(&meta("new")), &[], SHORT).unwrap_err().to_string();
        assert!(t.elapsed() >= SHORT.budget, "gave up after {:?}", t.elapsed());
        assert!(err.contains("another program holds") && err.contains("last.folo"), "{err}");
        drop(held);
        assert_eq!(std::fs::read(&path).unwrap(), bytes("old"));
        assert_eq!(names(&dir), ["last.folo"]);
    }
}
