//! Giving a finished file its name only while no file has that name ([`publish_new`], [`publish_free`]): how
//! exports and uploaded models are named without ever replacing a file.
//!
//! The file is written under a temporary name in the same folder ([`temp_path`]) and then moved to its name in one
//! step of the file system that fails when the name exists: `MoveFileExW` without `MOVEFILE_REPLACE_EXISTING` on
//! Windows, `renameat2` with `RENAME_NOREPLACE` on Linux. Nothing exists under the name until the file is complete,
//! so a writer that fails or is killed leaves the name free (or, killed after the move, the complete file); a killed
//! writer may leave its temporary file behind (empty, partial or complete), which nothing reads. Where that move does not exist (a Linux file
//! system without `RENAME_NOREPLACE`, macOS) a hard link is made instead, which is refused just the same when the
//! name exists, and the temporary name is then removed. A file system that refuses both is an `Unsupported` error:
//! the name is never claimed first and replaced afterwards.

use std::io;
use std::path::{Path, PathBuf};

/// The start of every [`temp_path`] name.
pub const TEMP_PREFIX: &str = ".folo-";

/// A temporary name in `dir` for a file to [`publish_new`] there: `.folo-<pid>-<n>.tmp`, at most 41 characters
/// whatever the final name, so a final name as long as a file system allows (255) still works, and it ends in
/// `.tmp` like every temporary file FOLO's folder scans skip. The file is created empty and left there for the writer
/// to write over: the name stays reserved, so another process with the same pid (another container sharing the
/// folder) never picks it too, and a missing or read-only folder fails here, before any work. The caller removes it
/// when it does not publish it.
pub fn temp_path(dir: &Path) -> io::Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    loop {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = dir.join(format!("{TEMP_PREFIX}{}-{n}.tmp", std::process::id()));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(_) => return Ok(path),
            // Another process's with the same pid, or a dead one's leftover.
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
}

/// Moves the finished file `tmp` to `path` (in the same folder, or at least on the same volume) only when no file
/// has that name, in one step: `AlreadyExists` when it is taken, and `tmp` is then left in place for the caller to
/// remove. See the module docs for the platforms.
pub fn publish_new(tmp: &Path, path: &Path) -> io::Result<()> {
    move_new(tmp, path)
}

/// [`publish_new`] under the first free name of `name(1)`, `name(2)`, ...; returns the name `tmp` took.
pub fn publish_free(tmp: &Path, mut name: impl FnMut(u32) -> PathBuf) -> io::Result<PathBuf> {
    let mut n = 1;
    loop {
        let p = name(n);
        match publish_new(tmp, &p) {
            Ok(()) => return Ok(p),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => n += 1,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(windows)]
fn move_new(from: &Path, to: &Path) -> io::Result<()> {
    use windows_sys::Win32::Storage::FileSystem::MoveFileExW;
    let (from, to) = (wide(from)?, wide(to)?);
    // No flags: not MOVEFILE_REPLACE_EXISTING (an existing `to` is ERROR_ALREADY_EXISTS), not MOVEFILE_COPY_ALLOWED
    // (a rename, never a copy and a delete).
    // SAFETY: both are NUL-terminated UTF-16 strings that live until the call returns; the result is checked.
    if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 0) } != 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

/// `p` as a NUL-terminated verbatim (`\\?\`) path, as `std::fs` passes long paths: `MoveFileExW` otherwise stops at
/// 260 characters. `std::path::absolute` first resolves `.`, `..` and `/` and drops trailing dots and spaces, as
/// Windows does for every other path; its error is returned. A NUL inside the path is `InvalidInput`, as in
/// `std::fs`.
#[cfg(windows)]
fn wide(p: &Path) -> io::Result<Vec<u16>> {
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Component, Prefix};
    let abs = std::path::absolute(p)?;
    if abs.as_os_str().encode_wide().any(|c| c == 0) {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "the path holds a NUL character"));
    }
    let (prefix, skip) = match abs.components().next() {
        Some(Component::Prefix(x)) => match x.kind() {
            Prefix::Disk(_) => (r"\\?\", 0),
            // `\\server\share\...` becomes `\\?\UNC\server\share\...`.
            Prefix::UNC(..) => (r"\\?\UNC", 1),
            // Verbatim or a device path already.
            _ => ("", 0),
        },
        _ => ("", 0),
    };
    Ok(prefix.encode_utf16().chain(abs.as_os_str().encode_wide().skip(skip)).chain(Some(0)).collect())
}

#[cfg(target_os = "linux")]
fn move_new(from: &Path, to: &Path) -> io::Result<()> {
    use std::ffi::{CString, c_char, c_int, c_uint};
    use std::os::unix::ffi::OsStrExt;
    unsafe extern "C" {
        // In glibc since 2.28 and musl since 1.2.
        fn renameat2(
            olddirfd: c_int,
            oldpath: *const c_char,
            newdirfd: c_int,
            newpath: *const c_char,
            flags: c_uint,
        ) -> c_int;
    }
    const AT_FDCWD: c_int = -100;
    const RENAME_NOREPLACE: c_uint = 1;
    let c = |p: &Path| CString::new(p.as_os_str().as_bytes()).map_err(io::Error::from);
    let (old, new) = (c(from)?, c(to)?);
    // SAFETY: both are NUL-terminated paths that live until the call returns; the result is checked.
    if unsafe { renameat2(AT_FDCWD, old.as_ptr(), AT_FDCWD, new.as_ptr(), RENAME_NOREPLACE) } == 0 {
        return Ok(());
    }
    let e = io::Error::last_os_error();
    match e.kind() {
        // No RENAME_NOREPLACE on this file system or kernel: EINVAL (what a file system without it answers), ENOSYS
        // and the other answers std calls `Unsupported`.
        io::ErrorKind::Unsupported | io::ErrorKind::InvalidInput => link_new(from, to),
        _ => Err(e),
    }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn move_new(from: &Path, to: &Path) -> io::Result<()> {
    link_new(from, to)
}

#[cfg(not(any(windows, unix)))]
fn move_new(_: &Path, to: &Path) -> io::Result<()> {
    Err(unsupported(to, io::Error::from(io::ErrorKind::Unsupported)))
}

/// A hard link `to` (refused when `to` exists), then `from` removed. A `from` that cannot be removed stays as a second
/// name of the published file.
#[cfg(unix)]
fn link_new(from: &Path, to: &Path) -> io::Result<()> {
    match std::fs::hard_link(from, to) {
        Ok(()) => {
            let _ = std::fs::remove_file(from);
            Ok(())
        }
        // No hard links on this file system.
        Err(e) if matches!(e.kind(), io::ErrorKind::Unsupported | io::ErrorKind::PermissionDenied) => {
            Err(unsupported(to, e))
        }
        Err(e) => Err(e),
    }
}

/// The error of a folder whose file system can publish `to` only at the risk of replacing a file.
#[cfg(not(windows))]
fn unsupported(to: &Path, e: io::Error) -> io::Error {
    let dir = to.parent().unwrap_or(Path::new("."));
    io::Error::new(
        io::ErrorKind::Unsupported,
        format!(
            "the file system of {} cannot give a file a name without risking an overwrite (no rename that refuses an \
             existing name, no hard link: {e})",
            dir.display()
        ),
    )
}

#[cfg(test)]
mod tests {
    use folo_test_support::TestDir;

    /// The names in `dir`, sorted.
    fn names(dir: &TestDir) -> Vec<String> {
        let mut n: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        n.sort();
        n
    }

    #[test]
    fn publishing_never_replaces_and_takes_the_next_free_name() {
        let dir = TestDir::new("nn-publish-free");
        let path = dir.join("m.onnx");
        let tmp = super::temp_path(dir.path()).unwrap();
        // The name stays reserved: an empty file the writer writes over.
        assert_eq!(std::fs::read(&tmp).unwrap(), b"");
        std::fs::write(&tmp, b"new").unwrap();
        // A file that appeared under the name just before the move keeps its bytes.
        std::fs::write(&path, b"old").unwrap();
        let err = super::publish_new(&tmp, &path).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        assert_eq!(std::fs::read(&tmp).unwrap(), b"new", "the temporary file stays for the caller");
        let name = |n: u32| dir.join(if n == 1 { "m.onnx".into() } else { format!("m-{n}.onnx") });
        assert_eq!(super::publish_free(&tmp, name).unwrap(), dir.join("m-2.onnx"));
        assert_eq!(std::fs::read(dir.join("m-2.onnx")).unwrap(), b"new");
        // One move: no second name of the published file stays behind.
        assert_eq!(names(&dir), ["m-2.onnx", "m.onnx"]);
    }

    #[test]
    fn a_name_as_long_as_the_file_system_allows_is_published() {
        let dir = TestDir::new("nn-publish-long");
        // 255 characters, the limit of one path component; the folder makes the whole path longer than 260.
        let name = format!("{}.onnx", "q".repeat(250));
        let path = dir.join(&name);
        let tmp = super::temp_path(dir.path()).unwrap();
        let tmp_name = tmp.file_name().unwrap().to_string_lossy().into_owned();
        assert!(tmp_name.starts_with(super::TEMP_PREFIX) && tmp_name.ends_with(".tmp"), "{tmp_name}");
        // The longest pid and counter.
        assert!(format!("{}{}-{}.tmp", super::TEMP_PREFIX, u32::MAX, u64::MAX).len() <= 41);
        std::fs::write(&tmp, b"done").unwrap();
        super::publish_new(&tmp, &path).unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"done");
        assert_eq!(names(&dir), [name]);
    }

    #[test]
    fn a_missing_folder_fails_before_any_work() {
        let dir = TestDir::new("nn-publish-missing");
        let err = super::temp_path(&dir.join("nope")).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }
}
