//! An optional, process-wide read guard. When a process installs one (`folo mcp --confine` does), every file or
//! folder this crate reads to load a dataset (manifests, image lists, split folders, COCO JSON, images, labels) is
//! passed to it first, and a refused path fails like an unreadable file. Without a guard, nothing changes.

use std::path::Path;
use std::sync::OnceLock;

/// Part of the message of every refused read, so callers can recognise it.
pub const DENIED: &str = "is outside the folders this process may read";

type Allow = Box<dyn Fn(&Path) -> bool + Send + Sync>;

static GUARD: OnceLock<Allow> = OnceLock::new();

/// Installs `allow` for the rest of the process. Returns false (and keeps the first) if one is already installed.
pub fn install(allow: impl Fn(&Path) -> bool + Send + Sync + 'static) -> bool {
    GUARD.set(Box::new(allow)).is_ok()
}

/// `Ok` when `path` may be read (always without a guard), else a `PermissionDenied` error naming it.
pub fn check(path: &Path) -> std::io::Result<()> {
    match GUARD.get() {
        Some(allow) if !allow(path) => Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{} {DENIED}", crate::paths::shown(path)),
        )),
        _ => Ok(()),
    }
}

/// True when a guard is installed (callers then re-read what they would otherwise take from a cache).
pub fn installed() -> bool {
    GUARD.get().is_some()
}

/// True when `path` may be read.
pub fn allowed(path: &Path) -> bool {
    check(path).is_ok()
}
