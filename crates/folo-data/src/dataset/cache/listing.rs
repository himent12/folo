//! Folder listings: whether a file exists and its [`Stamp`] for thousands of files from one directory read, instead
//! of one `stat` (an open and close on Windows) per file.

use super::{Settings, Stamp};
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// What a listing (or a `stat`) says about one path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    /// A regular file, links followed (what [`Path::is_file`] says).
    pub is_file: bool,
    /// Its length and modification time, when known.
    pub stamp: Option<Stamp>,
}

impl Entry {
    fn of(m: &std::fs::Metadata) -> Self {
        Entry { is_file: m.is_file(), stamp: Stamp::of(m) }
    }
}

/// The listed folders of one open.
#[derive(Default)]
pub(crate) struct Listing {
    dirs: HashMap<PathBuf, HashMap<OsString, Entry>>,
}

impl Listing {
    /// Lists, in parallel, every folder that holds at least `settings.min_files` of `files`. A folder that cannot
    /// be read is left out (its files are then looked at one by one, as without a listing).
    pub fn new<'a>(files: impl Iterator<Item = &'a Path>, settings: &Settings) -> Self {
        if !settings.list {
            return Self::default();
        }
        let mut counts: HashMap<&Path, usize> = HashMap::new();
        for f in files {
            if let Some(d) = f.parent() {
                *counts.entry(d).or_default() += 1;
            }
        }
        let wanted: Vec<&Path> =
            counts.into_iter().filter(|&(_, n)| n >= settings.min_files.max(1)).map(|(d, _)| d).collect();
        let dirs = wanted.into_par_iter().filter_map(|d| Some((d.to_path_buf(), list(d)?))).collect();
        Listing { dirs }
    }

    /// The listed folders.
    pub fn dirs(&self) -> impl Iterator<Item = &Path> {
        self.dirs.keys().map(PathBuf::as_path)
    }

    /// What is at `path`: from its folder's listing when the folder was listed and has the name as a plain file
    /// or folder, else from the file system (a link, a name in another case on a case-insensitive disk, a missing
    /// file). `None` when nothing is there.
    pub fn stat(&self, path: &Path) -> Option<Entry> {
        if let (Some(dir), Some(name)) = (path.parent(), path.file_name())
            && let Some(e) = self.dirs.get(dir).and_then(|m| m.get(name))
        {
            return Some(*e);
        }
        std::fs::metadata(path).ok().map(|m| Entry::of(&m))
    }
}

/// The plain files and folders of `dir` (links are left out: [`Listing::stat`] follows them itself).
fn list(dir: &Path) -> Option<HashMap<OsString, Entry>> {
    let mut out = HashMap::new();
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let Ok(ty) = e.file_type() else { continue };
        if ty.is_symlink() {
            continue;
        }
        // Free on Windows (the directory read returns it); one `statx` per entry elsewhere.
        if let Ok(m) = e.metadata() {
            out.insert(e.file_name(), Entry::of(&m));
        }
    }
    Some(out)
}
