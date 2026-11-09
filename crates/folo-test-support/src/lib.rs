//! Test helpers shared by FOLO's crates (a dev-dependency only).
//!
//! Every test that needs a scratch folder takes a [`TestDir`]: a fresh `folo-<tag>-...` directory under [`root`]
//! (`FOLO_TEST_TMP` when set, else the system temp dir) that is removed when the guard drops, also when the test
//! panics. A test that spawns a process (`folo serve`, `folo train`, `folo mcp`) wraps it in a [`ChildGuard`],
//! declared *after* the directory it uses, so the process is killed before the directory is removed (on Windows a
//! file that is still open cannot be deleted).
//!
//! The workspace test `tests/hygiene.rs` keeps it that way: no test or test helper outside this crate may call
//! `std::env::temp_dir()` itself.

use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::process::Child;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The environment variable naming the folder test directories go in (default: the system temp dir).
pub const ROOT_ENV: &str = "FOLO_TEST_TMP";

/// The folder test directories go in: `FOLO_TEST_TMP` when it is set and not empty (created if missing), else
/// [`std::env::temp_dir`]. Panics when `FOLO_TEST_TMP` is relative: cargo runs each test binary in its own package
/// folder, so a relative path would scatter test folders across the workspace.
pub fn root() -> PathBuf {
    match std::env::var_os(ROOT_ENV) {
        Some(p) if !p.is_empty() => {
            let p = PathBuf::from(p);
            assert!(p.is_absolute(), "{ROOT_ENV}={} must be an absolute path", p.display());
            std::fs::create_dir_all(&p).unwrap_or_else(|e| panic!("{ROOT_ENV}={}: {e}", p.display()));
            p
        }
        _ => std::env::temp_dir(),
    }
}

/// A fresh, empty directory under [`root`], removed with everything in it when dropped.
#[derive(Debug)]
pub struct TestDir {
    path: PathBuf,
}

impl TestDir {
    /// Creates `<root>/folo-<tag>-<pid>-<n>`, unique within and across test processes.
    pub fn new(tag: &str) -> TestDir {
        static N: AtomicU64 = AtomicU64::new(0);
        let root = root();
        loop {
            let n = N.fetch_add(1, Ordering::Relaxed);
            let path = root.join(format!("folo-{tag}-{}-{n}", std::process::id()));
            match std::fs::create_dir(&path) {
                Ok(()) => return TestDir { path },
                // A folder a crashed run with the same pid left behind: take the next number.
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("cannot create {}: {e}", path.display()),
            }
        }
    }

    /// A guard for a fresh directory at `path`, for the rare test that needs one outside [`root`] (say, under the
    /// working directory, to reach it by a relative path): a leftover from an earlier run is removed first.
    pub fn at(path: impl Into<PathBuf>) -> TestDir {
        let path = path.into();
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap_or_else(|e| panic!("cannot create {}: {e}", path.display()));
        TestDir { path }
    }

    /// The directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Deref for TestDir {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TestDir {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        // Retried for a few seconds: on Windows a file stays locked briefly after the process that had it open
        // exits, and a virus scanner may hold a fresh file. Never panics (this runs while a failed test unwinds).
        let mut last = None;
        for attempt in 0..40 {
            match std::fs::remove_dir_all(&self.path) {
                Ok(()) => return,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => return,
                Err(e) => last = Some(e),
            }
            std::thread::sleep(Duration::from_millis(if attempt < 10 { 20 } else { 100 }));
        }
        if let Some(e) = last {
            eprintln!("warning: could not remove test dir {}: {e}", self.path.display());
        }
    }
}

/// The path of a file `name` in its own fresh [`TestDir`] (the file is not created); the folder goes when this drops.
#[derive(Debug)]
pub struct TestFile {
    path: PathBuf,
    _dir: TestDir,
}

impl TestFile {
    /// `<root>/folo-<tag>-<pid>-<n>/<name>`.
    pub fn new(tag: &str, name: &str) -> TestFile {
        let dir = TestDir::new(tag);
        TestFile { path: dir.join(name), _dir: dir }
    }

    /// The file's path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Deref for TestFile {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for TestFile {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

/// A spawned process that is killed (and reaped) when the guard drops, unless it has exited by then.
#[derive(Debug)]
pub struct ChildGuard(Child);

impl ChildGuard {
    /// Takes ownership of `child`.
    pub fn new(child: Child) -> ChildGuard {
        ChildGuard(child)
    }
}

impl From<Child> for ChildGuard {
    fn from(child: Child) -> ChildGuard {
        ChildGuard(child)
    }
}

impl Deref for ChildGuard {
    type Target = Child;

    fn deref(&self) -> &Child {
        &self.0
    }
}

impl DerefMut for ChildGuard {
    fn deref_mut(&mut self) -> &mut Child {
        &mut self.0
    }
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dirs_are_unique_and_removed_on_drop() {
        let (a, b) = (TestDir::new("support"), TestDir::new("support"));
        assert_ne!(a.path(), b.path());
        let name = a.file_name().unwrap().to_str().unwrap().to_string();
        assert!(name.starts_with("folo-support-"), "{name}");
        std::fs::create_dir_all(a.join("x/y")).unwrap();
        std::fs::write(a.join("x/y/z.txt"), b"z").unwrap();
        let p = a.path().to_path_buf();
        drop(a);
        assert!(!p.exists());
        assert!(b.exists());
    }

    #[test]
    fn test_dirs_are_removed_when_a_test_panics() {
        let mut path = None;
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let d = TestDir::new("support-panic");
            std::fs::write(d.join("f"), b"f").unwrap();
            path = Some(d.path().to_path_buf());
            panic!("boom");
        }));
        assert!(r.is_err());
        assert!(!path.unwrap().exists());
    }

    #[test]
    fn child_guard_kills_a_running_process() {
        let mut cmd = if cfg!(windows) {
            let mut c = std::process::Command::new("ping");
            c.args(["-n", "30", "127.0.0.1"]);
            c
        } else {
            let mut c = std::process::Command::new("sleep");
            c.arg("30");
            c
        };
        let child = cmd.stdout(std::process::Stdio::null()).spawn().unwrap();
        let id = child.id();
        let start = std::time::Instant::now();
        drop(ChildGuard::new(child));
        assert!(start.elapsed() < Duration::from_secs(10), "pid {id} was waited for, not killed");
    }
}
