//! Streaming tar archive extraction, path validation, case-folding collision detection, and partial-file cleanup.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::error::{Error, Result};

use super::pipe::{Pipe, PipeReader};
use super::{invalid, io_error, truncated};

/// An error of the tar reader: the stream is not a readable archive.
fn unreadable(e: std::io::Error) -> Error {
    invalid(format!("not a readable tar archive: {e}"))
}

/// Unpacks the archive arriving through `pipe` into `root`, then reads the stream to its end (writers pad archives
/// after the end-of-archive marker). The stream ends only once the marker arrived ([`Pipe::finish`]), so the reader
/// never runs out inside the archive.
pub(super) fn unpack(pipe: &Arc<Pipe>, root: &Path) -> Result<()> {
    let mut archive = tar::Archive::new(PipeReader(pipe.clone()));
    let mut buf = vec![0u8; 64 * 1024];
    // Each file's name as Windows compares it ([`folded`]) -> the entry's name.
    let mut files: HashMap<String, String> = HashMap::new();
    for entry in archive.entries().map_err(unreadable)? {
        let mut entry = entry.map_err(unreadable)?;
        let kind = entry.header().entry_type();
        if kind.is_pax_global_extensions() {
            // Archive-wide metadata (`git archive` writes one), not a file.
            continue;
        }
        let rel = entry_path(&entry)?;
        let path = root.join(&rel);
        if !path.starts_with(root) {
            return Err(invalid(format!(
                "{}: only relative paths inside the dataset folder are accepted",
                rel.display()
            )));
        }
        if let Some(clash) = file_folder_clash(root, &rel, kind.is_dir()) {
            return Err(invalid(format!(
                "{clash} is both a file and a folder in the archive; rename one and pack again"
            )));
        }
        if kind.is_dir() {
            std::fs::create_dir_all(&path).map_err(|e| io_error("creating", &path, e))?;
        } else if kind.is_file() || kind.is_contiguous() {
            if rel.as_os_str().is_empty() {
                return Err(invalid("a file entry without a name"));
            }
            let name = folo_api::json::path(&rel);
            if let Some(other) = files.insert(folded(&rel), name.clone())
                && other != name
            {
                return Err(invalid(format!(
                    "{other} and {name} differ only by letter case or a trailing dot or space, so one would \
                     overwrite the other on Windows; rename one (and the labels that refer to it) and pack again"
                )));
            }
            pipe.reading(&name);
            write_entry(&mut entry, &path, &name, &mut buf)?;
            pipe.read();
        } else {
            let what = if kind.is_symlink() {
                "a symbolic link"
            } else if kind.is_hard_link() {
                "a hard link"
            } else {
                "a special file (device, FIFO, sparse file)"
            };
            return Err(invalid(format!("{} is {what}; only files and folders are accepted", rel.display())));
        }
    }
    pipe.end();
    std::io::copy(&mut archive.into_inner(), &mut std::io::sink()).map_err(unreadable)?;
    Ok(())
}

/// The relative path an entry unpacks to: its normal components (`.` skipped). An absolute path, `..`, a drive,
/// a `:` (an NTFS stream), a NUL byte, or a backslash inside a component is `422 invalid_upload` on every OS; on
/// Windows, components that fail to unpack or lose data there (`< > " | ? *`, ASCII control bytes `< 0x20`,
/// dot/space-only names, `NUL` with optional trailing dots/spaces, or a parent folder ending in a space or more
/// than one dot/space) are `422 invalid_upload` too.
pub(super) fn entry_path(entry: &tar::Entry<impl Read>) -> Result<PathBuf> {
    let raw = entry.path().map_err(|e| invalid(format!("an entry's name is not a path: {e}")))?;
    let mut parts = Vec::new();
    for c in raw.components() {
        match c {
            Component::CurDir => {}
            Component::Normal(n) => {
                let s = n.to_string_lossy();
                if s.contains(':') || s.contains('\\') || s.contains('\0') {
                    return Err(invalid(format!(
                        "{}: only relative paths inside the dataset folder are accepted",
                        raw.display()
                    )));
                }
                parts.push(n.to_os_string());
            }
            _ => {
                return Err(invalid(format!(
                    "{}: only relative paths inside the dataset folder are accepted",
                    raw.display()
                )));
            }
        }
    }
    #[cfg(windows)]
    for (i, n) in parts.iter().enumerate() {
        let s = n.to_string_lossy();
        let is_parent = i + 1 < parts.len();
        if let Some(why) = windows_unsafe_component(&s, is_parent) {
            return Err(invalid(format!("{}: {why}; rename it and pack again", raw.display())));
        }
    }
    Ok(parts.into_iter().collect())
}

/// Why a path component fails to unpack or loses data on Windows (`os error 123`/`5`/`3` or writing to
/// `\Device\Null`). Narrowed to what fails on Windows 11 with `std::fs`: forbidden characters (`< > " | ? *` and
/// ASCII control bytes `< 0x20`), dot/space-only components (`...`, `   `), bare `NUL` (with optional trailing
/// dots/spaces), and parent folders ending in a space or multiple dots/spaces (`foo /a.png`, `foo../a.png`).
/// Names that unpack cleanly on Windows 11 (`CON.png`, `NUL.png`, `aux/`, bare `CON`/`AUX`/`PRN`/`COM*`/`LPT*`,
/// and files with a trailing dot or space) stay accepted; collisions remain caught by [`folded`].
#[cfg(windows)]
fn windows_unsafe_component(name: &str, is_parent: bool) -> Option<&'static str> {
    if name.bytes().any(|b| b < 0x20 || matches!(b, b'<' | b'>' | b'"' | b'|' | b'?' | b'*')) {
        return Some("contains characters Windows forbids in file names");
    }
    let trimmed = name.trim_end_matches(['.', ' ']);
    if trimmed.is_empty() {
        return Some("is empty on Windows (only dots or spaces)");
    }
    if trimmed.eq_ignore_ascii_case("nul") {
        return Some("uses the reserved Windows device name NUL");
    }
    if is_parent && trimmed != name && !name.strip_suffix('.').is_some_and(|s| s == trimmed) {
        return Some(
            "has a folder name ending in spaces or multiple dots that Windows cannot open files under",
        );
    }
    None
}

/// The name of what an entry at `rel` (a folder when `dir`) collides with in `root`: an earlier file where it or one
/// of its parent folders goes, or an earlier folder where a file goes. Without this check the unpacking fails with an
/// operating system error (`500 io_error`) instead of refusing the archive.
fn file_folder_clash(root: &Path, rel: &Path, dir: bool) -> Option<String> {
    let mut at = root.to_path_buf();
    let parts: Vec<_> = rel.components().collect();
    for (i, c) in parts.iter().enumerate() {
        at.push(c);
        let last = i + 1 == parts.len();
        let Ok(meta) = std::fs::symlink_metadata(&at) else { return None };
        let wants_dir = !last || dir;
        if wants_dir != meta.is_dir() {
            let name: PathBuf = parts[..=i].iter().collect();
            return Some(folo_api::json::path(&name));
        }
    }
    None
}

/// `rel` as Windows compares names: each component without trailing dots and spaces, in lower case.
fn folded(rel: &Path) -> String {
    let parts: Vec<String> = rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy().trim_end_matches(['.', ' ']).to_lowercase())
        .collect();
    parts.join("/")
}

/// Writes file entry `name` to `path`, removing the partial file if reading or writing fails mid-entry.
pub(super) fn write_entry(
    entry: &mut tar::Entry<impl Read>,
    path: &Path,
    name: &str,
    buf: &mut [u8],
) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| io_error("creating", dir, e))?;
    }
    let file = std::fs::File::create(path).map_err(|e| io_error("creating", path, e))?;
    let res = copy_entry_bytes(entry, file, path, name, buf);
    if res.is_err() {
        // `file` was moved into `copy_entry_bytes` and closed before we remove the partial file (required on Windows).
        let _ = std::fs::remove_file(path);
    }
    res
}

fn copy_entry_bytes(
    entry: &mut tar::Entry<impl Read>,
    mut file: std::fs::File,
    path: &Path,
    name: &str,
    buf: &mut [u8],
) -> Result<()> {
    let mut left = entry.size();
    while left > 0 {
        let n = entry.read(buf).map_err(unreadable)?;
        if n == 0 {
            return Err(truncated(format!("the stream ends inside {name}")));
        }
        file.write_all(&buf[..n]).map_err(|e| io_error("writing", path, e))?;
        left = left.saturating_sub(n as u64);
    }
    Ok(())
}
