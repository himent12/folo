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

    /// 10 ms, doubling, at most 50 ms per wait and 250 ms in all: for small state files written while the caller
    /// holds the lock of the state they persist (a server's `run.json`), where a long wait would stall everything
    /// that waits for that lock. Such a write that fails is logged and made again with the next change.
    pub const BRIEF: RenameRetry = RenameRetry {
        first: Duration::from_millis(10),
        cap: Duration::from_millis(50),
        budget: Duration::from_millis(250),
    };
}

/// Renames `tmp` over `path` ([`rename_over_with`] with [`RenameRetry::DEFAULT`]). The caller removes `tmp` when
/// this fails.
pub fn rename_over(tmp: &Path, path: &Path) -> io::Result<()> {
    rename_over_with(tmp, path, RenameRetry::DEFAULT).map(drop)
}

/// The shortest wait between attempts, whatever [`RenameRetry`] says: a zero wait would spin.
const MIN_WAIT: Duration = Duration::from_millis(1);

/// Renames `tmp` over `path`, retrying while another program holds `path` open (Windows only, see the module
/// docs) for at most `retry.budget`, and returns how long it waited (zero when the first attempt succeeded). Any
/// other error is returned at once. The error after the budget names `path`, says another program may hold it
/// open or its permissions may not allow replacing it (Windows gives the same error for both), and keeps the
/// error of the last attempt. `tmp` is left in place either way: the caller removes it when this fails.
pub fn rename_over_with(tmp: &Path, path: &Path, retry: RenameRetry) -> io::Result<Duration> {
    rename_over_observed(tmp, path, retry, &mut || {})
}

/// [`rename_over_with`], calling `retrying` after every failed attempt that is tried again, before the wait (tests
/// let go of the file there, so they know a first attempt failed).
fn rename_over_observed(
    tmp: &Path,
    path: &Path,
    retry: RenameRetry,
    retrying: &mut dyn FnMut(),
) -> io::Result<Duration> {
    let start = Instant::now();
    let cap = retry.cap.max(MIN_WAIT);
    let mut wait = retry.first.clamp(MIN_WAIT, cap);
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
                    "cannot replace {}: another program may hold it open, or its permissions may not allow \
                     replacing it (retried for {:.1} s; close any program that has the file open, or check the \
                     permissions of the file and its folder): {err}",
                    path.display(),
                    spent.as_secs_f64()
                ),
            ));
        }
        retrying();
        let nap = wait.min(retry.budget - spent);
        std::thread::sleep(nap);
        waited += nap;
        wait = wait.saturating_mul(2).min(cap);
    }
}

/// Whether `e`, from renaming over `path`, may mean another program holds `path` open (and may let go). Access
/// denied is also what a directory or a read-only file at `path` gives: those fail at once. A file whose
/// permissions (its ACL, or its folder's) do not allow replacing it gives the same error as a held one and cannot
/// be told apart without a race, so it is retried for the budget too, and the final error mentions both.
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
        // Counted, not timed: a loaded machine can take a second to write and sync the temporary file.
        let tmp = dir.join(".last.folo.tmp");
        std::fs::write(&tmp, b"new").unwrap();
        let slow = RenameRetry { budget: Duration::from_secs(5), ..SHORT };
        let mut retries = 0;
        assert!(super::rename_over_observed(&tmp, &path, slow, &mut || retries += 1).is_err());
        assert_eq!(retries, 0, "a permanent error was tried again");
        std::fs::remove_file(&tmp).unwrap();
        assert!(write_file_with(&path, Some(&meta("new")), &[], slow).is_err());
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

    /// A reader that lets go only once an attempt has failed (so a plain rename, which never tries again, fails
    /// this every time, however the threads are scheduled).
    #[cfg(windows)]
    #[test]
    fn a_rename_waits_for_a_reader_that_lets_go() {
        let dir = TestDir::new("nn-replace-reader");
        let (path, tmp) = (dir.join("last.folo"), dir.join(".last.folo.tmp"));
        std::fs::write(&path, b"old").unwrap();
        std::fs::write(&tmp, b"new").unwrap();
        let mut held = Some(hold(&path));
        let mut retries = 0;
        super::rename_over_observed(&tmp, &path, SHORT, &mut || {
            retries += 1;
            held = None;
        })
        .unwrap();
        assert!(retries >= 1 && held.is_none(), "the first attempt, with the file held, did not fail");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
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
        let (done, result) = std::sync::mpsc::channel();
        let p = path.clone();
        std::thread::spawn(move || {
            let _ = done.send(write_file_with(&p, Some(&meta("new")), &[], SHORT).map_err(|e| e.to_string()));
        });
        // A watchdog far above the 300 ms budget: a loop that never gives up fails here rather than hanging.
        let err =
            result.recv_timeout(Duration::from_secs(30)).expect("the retry gave up within 30 s").unwrap_err();
        assert!(t.elapsed() >= SHORT.budget, "gave up after {:?}", t.elapsed());
        assert!(err.contains("another program may hold it open") && err.contains("last.folo"), "{err}");
        drop(held);
        assert_eq!(std::fs::read(&path).unwrap(), bytes("old"));
        assert_eq!(names(&dir), ["last.folo"]);
    }

    /// Denies everyone (`*S-1-1-0`) the rights to `paths` that replacing a file needs, until dropped.
    #[cfg(windows)]
    struct Deny(Vec<std::path::PathBuf>);

    #[cfg(windows)]
    impl Deny {
        fn new(rules: &[(&std::path::Path, &str)]) -> Deny {
            let mut deny = Deny(Vec::new());
            for (p, rights) in rules {
                let s = std::process::Command::new("icacls")
                    .arg(p)
                    .args(["/deny", &format!("*S-1-1-0:{rights}")])
                    .output()
                    .unwrap();
                assert!(s.status.success(), "icacls /deny: {}", String::from_utf8_lossy(&s.stdout));
                deny.0.push(p.to_path_buf());
            }
            deny
        }
    }

    #[cfg(windows)]
    impl Drop for Deny {
        fn drop(&mut self) {
            for p in &self.0 {
                let _ = std::process::Command::new("icacls").arg(p).args(["/remove:d", "*S-1-1-0"]).output();
            }
        }
    }

    /// A target whose permissions deny replacing it fails with the same error as a held one: the final error says
    /// it may be either, never that it is not a permissions problem.
    #[cfg(windows)]
    #[test]
    fn a_target_whose_permissions_deny_replacing_it_says_permissions_may_be_why() {
        let dir = TestDir::new("nn-replace-acl");
        let path = dir.join("last.folo");
        write_file(&path, Some(&meta("old")), &[]).unwrap();
        let deny = Deny::new(&[(dir.path(), "(DC)"), (&path, "(DE)")]);
        let err = write_file_with(&path, Some(&meta("new")), &[], SHORT).unwrap_err().to_string();
        drop(deny);
        assert!(err.contains("its permissions may not allow replacing it"), "{err}");
        assert!(!err.contains("not a permissions problem"), "{err}");
        assert_eq!(std::fs::read(&path).unwrap(), bytes("old"));
        assert_eq!(names(&dir), ["last.folo"]);
    }

    /// Zero waits do not spin (at least 1 ms between attempts), and huge ones do not overflow when doubled.
    #[cfg(windows)]
    #[test]
    fn retry_settings_out_of_range_neither_spin_nor_panic() {
        let dir = TestDir::new("nn-replace-range");
        let (path, tmp) = (dir.join("last.folo"), dir.join(".last.folo.tmp"));
        std::fs::write(&path, b"old").unwrap();
        std::fs::write(&tmp, b"new").unwrap();
        let held = hold(&path);
        let zero =
            RenameRetry { first: Duration::ZERO, cap: Duration::ZERO, budget: Duration::from_millis(50) };
        let mut attempts = 0;
        assert!(super::rename_over_observed(&tmp, &path, zero, &mut || attempts += 1).is_err());
        assert!(attempts <= 60, "{attempts} attempts in 50 ms");
        let huge =
            RenameRetry { first: Duration::MAX, cap: Duration::MAX, budget: Duration::from_millis(30) };
        assert!(super::rename_over_with(&tmp, &path, huge).is_err());
        drop(held);
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
    }
}
