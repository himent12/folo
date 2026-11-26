//! Datasets on disk, read with folo-data: an [`Index`] of every image (file, split, size, pixel boxes) plus the
//! dataset statistics. [`RealStore`](crate::RealStore) caches indexes as JSON and rebuilds one when its
//! [`Fingerprint`] (file sizes and modification times under the dataset folder) changes; thumbnails are cached as
//! JPEG files keyed by the image's size and modification time.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use axum::http::StatusCode;
use folo_api::{
    BBox, DatasetFormat, DatasetIssue, DatasetStats, Histogram, ImageInfo, ImagePage, IssueSeverity, Split,
    Splits,
};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::store::{ImageData, ImageQuery, IssueQuery};
use crate::util::fnv1a;

/// Most classes a dataset may have.
pub const MAX_CLASSES: usize = 4096;

/// Bumped whenever the cached [`Index`] layout or its derivation changes, so old caches are rebuilt. 6: empty YOLO
/// split paths are unset. 7: folo-data reads COCO splits, float category ids and crowd flags, and labels below a
/// folder called `images`, differently.
pub const INDEX_VERSION: u32 = 7;

/// A cheap summary of every file under a dataset's folders: count, total size and a hash of each file's relative
/// path, size and modification time (and of each folder's name). Adding, removing, renaming or editing a file, or
/// creating or removing a folder, changes it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fingerprint {
    /// Files seen.
    pub files: u64,
    /// Their total size in bytes.
    pub bytes: u64,
    /// FNV-1a over `(path, size, mtime)` of each file, in sorted path order (hex).
    pub hash: String,
}

impl Fingerprint {
    /// Walks `dirs` (folders recursively, following links; files are hashed relative to their walk root). A
    /// path may also be a single file or not exist. Paths under `exclude` (the server's own state, when its data
    /// dir sits inside a dataset folder) are skipped, and so are the temporary files [`write_atomic`] makes for
    /// them.
    pub fn of(dirs: &[PathBuf], exclude: &[PathBuf]) -> Fingerprint {
        let mut files = 0;
        let mut bytes = 0;
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |data: &[u8]| {
            for &b in data {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for dir in dirs {
            feed(dir.to_string_lossy().as_bytes());
            // A missing path is part of the fingerprint too: creating it must count as a change.
            feed(&[u8::from(dir.exists())]);
            let walk = walkdir::WalkDir::new(dir).follow_links(true).sort_by_file_name();
            let walk = walk.into_iter().filter_entry(|e| !excluded(e.path(), exclude));
            for e in walk.filter_map(|e| e.ok()) {
                if e.file_type().is_dir() {
                    // Folders count by name: creating an empty one can change the dataset (a `path:` or split
                    // folder that did not exist yet).
                    let rel = e.path().strip_prefix(dir).unwrap_or(e.path());
                    feed(rel.to_string_lossy().as_bytes());
                    feed(b"/");
                    continue;
                }
                if !e.file_type().is_file() {
                    continue;
                }
                let Ok(md) = e.metadata() else { continue };
                let mtime = md
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_nanos());
                let rel = e.path().strip_prefix(dir).unwrap_or(e.path());
                feed(rel.to_string_lossy().as_bytes());
                feed(&md.len().to_le_bytes());
                feed(&mtime.to_le_bytes());
                files += 1;
                bytes += md.len();
            }
        }
        Fingerprint { files, bytes, hash: format!("{h:016x}") }
    }
}

/// True when `p` is under one of `exclude`, or is a [`write_atomic`] temporary file of one.
fn excluded(p: &Path, exclude: &[PathBuf]) -> bool {
    let p = lexical(p);
    exclude.iter().any(|x| {
        if p.starts_with(x) {
            return true;
        }
        let (Some(dir), Some(name)) = (x.parent(), x.file_name()) else { return false };
        let prefix = format!(".{}.", name.to_string_lossy());
        p.parent() == Some(dir)
            && p.file_name()
                .map(|n| n.to_string_lossy())
                .is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".tmp"))
    })
}

/// One image of an [`Index`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexImage {
    /// Path relative to [`Index::root`] with `/` separators (absolute when the image lies outside it), as
    /// [`folo_data::paths::relative_to`] names it.
    pub file: String,
    /// Split.
    pub split: Split,
    /// Width in pixels (after EXIF orientation).
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Ground truth in pixel `xyxy`.
    pub boxes: Vec<BBox>,
}

/// Everything the API serves about a dataset, built once with folo-data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Index {
    /// [`INDEX_VERSION`] when built.
    pub version: u32,
    /// The files the index was built from.
    pub fingerprint: Fingerprint,
    /// The folders the fingerprint covers.
    pub watched: Vec<PathBuf>,
    /// Paths the fingerprint skips (see [`Fingerprint::of`]).
    pub excluded: Vec<PathBuf>,
    /// The registered folder (or the folder of a registered file).
    pub base: PathBuf,
    /// The dataset root as folo-data resolves it (a `data.yaml`'s `path:`, the folder above `annotations/`):
    /// image and issue `file`s are relative to it, exactly like `folo dataset check --json`.
    pub root: PathBuf,
    /// Format that was read.
    pub format: DatasetFormat,
    /// Class names.
    pub classes: Vec<String>,
    /// Images per split.
    pub splits: Splits,
    /// Label statistics.
    pub stats: DatasetStats,
    /// Images in split order, then path order; the image id is the position.
    pub images: Vec<IndexImage>,
    /// What `folo dataset check --headers-only` finds, files relative to `root`.
    pub issues: Vec<DatasetIssue>,
}

fn data_format(f: DatasetFormat) -> folo_data::Format {
    match f {
        DatasetFormat::Auto => folo_data::Format::Auto,
        DatasetFormat::Yolo => folo_data::Format::Yolo,
        DatasetFormat::Coco => folo_data::Format::Coco,
    }
}

fn api_split(s: folo_data::Split) -> Split {
    match s {
        folo_data::Split::Train => Split::Train,
        folo_data::Split::Val => Split::Val,
        folo_data::Split::Test => Split::Test,
    }
}

fn api_hist(h: folo_data::Histogram) -> Histogram {
    Histogram { edges: h.edges, counts: h.counts }
}

/// The API stats object of folo-data statistics.
pub fn api_stats(s: folo_data::DatasetStats) -> DatasetStats {
    DatasetStats {
        boxes: s.boxes,
        empty_images: s.empty_images,
        broken_labels: s.broken_labels,
        class_counts: s.class_counts,
        box_size_hist: api_hist(s.box_size_hist),
        boxes_per_image_hist: api_hist(s.boxes_per_image_hist),
    }
}

/// Maps a folo-data error on dataset `path` to the API's error codes.
pub fn data_error(e: folo_data::DataError, path: &str) -> Error {
    use folo_data::DataError;
    match e {
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            let path = folo_data::paths::shown(path);
            Error::new(StatusCode::NOT_FOUND, "path_not_found", format!("'{path}' does not exist")).hint(
                "pass the dataset root: the folder holding data.yaml, images/ + labels/, or annotations/",
            )
        }
        // The message already says what a dataset folder holds. A dataset one or two folders down (the folder above
        // it was picked, or an archive wraps it in a folder) is named; images there without labels are said as that.
        DataError::Format(m) if m.starts_with(folo_data::UNDETECTED) => {
            if let Some(below) = dataset_below(Path::new(path), DIAGNOSE_ENTRIES) {
                let (shown, below) = (folo_data::paths::shown(path), folo_data::paths::shown(below));
                return Error::new(
                    StatusCode::BAD_REQUEST,
                    "unknown_format",
                    format!("'{shown}' is not a dataset folder, but '{below}' inside it holds a dataset"),
                )
                .hint(format!(
                    "register '{below}': the folder holding data.yaml, images/ + labels/, or annotations/"
                ));
            }
            match images_under(Path::new(path), DIAGNOSE_ENTRIES) {
                0 => Error::new(StatusCode::BAD_REQUEST, "unknown_format", m),
                n => {
                    let (path, s) = (folo_data::paths::shown(path), if n == 1 { "" } else { "s" });
                    let message = format!("'{path}' has {n} image{s} but no labels");
                    Error::new(StatusCode::BAD_REQUEST, "unknown_format", message).hint(
                        "label them first (FOLO Studio: Datasets, Label images), or add YOLO labels/ or a COCO \
                         annotations/ JSON",
                    )
                }
            }
        }
        DataError::Format(m) => Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", m),
        DataError::Unfinished(_) => Error::new(StatusCode::BAD_REQUEST, "dataset_unfinished", e.to_string())
            .hint(folo_data::unfinished::HINT),
        DataError::Config(m) => Error::new(StatusCode::BAD_REQUEST, "invalid_body", m),
        e => Error::internal(e.to_string()),
    }
}

/// Entries the diagnostics of a folder that is not a dataset look at, at most ([`dataset_below`], [`images_under`]):
/// they only phrase an error, so a drive or a system folder is never walked to say it is not one (x3-server-1).
pub(crate) const DIAGNOSE_ENTRIES: usize = 20_000;

/// The first folder one or two levels below `dir` (in name order, at most 64 folders a level, `budget` folder entries
/// read in all, also those [`looks_like_dataset`] reads) where a dataset is found, for a folder that is not a dataset
/// itself.
pub(crate) fn dataset_below(dir: &Path, mut budget: usize) -> Option<PathBuf> {
    fn folders(dir: &Path, budget: &mut usize) -> Vec<PathBuf> {
        // A folder the read guard refuses (`folo mcp --confine`) is not listed.
        if folo_data::guard::check(dir).is_err() {
            return Vec::new();
        }
        let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
        let mut dirs = Vec::new();
        for e in entries {
            if *budget == 0 || dirs.len() == 64 {
                break;
            }
            *budget -= 1;
            if let Ok(e) = e
                && e.path().is_dir()
            {
                dirs.push(e.path());
            }
        }
        dirs.sort();
        dirs
    }
    let first = folders(dir, &mut budget);
    if let Some(d) = first.iter().find(|d| looks_like_dataset(d, &mut budget)) {
        return Some(d.clone());
    }
    for d in &first {
        if let Some(found) = folders(d, &mut budget).into_iter().find(|d| looks_like_dataset(d, &mut budget))
        {
            return Some(found);
        }
    }
    None
}

/// Whether `dir` holds what [`folo_data::detect_format`] finds a dataset by, looked at within `budget` (one entry
/// per path checked or folder entry read, a YAML read up to 64 KiB): `detect_format` reads every entry of the
/// folder and its `annotations/`, which a folder of millions of files makes unbounded (x3-server-1, review round 2).
/// It only phrases an error, so a dataset whose YAML is beyond the budget is missed, never walked to.
fn looks_like_dataset(dir: &Path, budget: &mut usize) -> bool {
    let mut charge = |n: usize| -> bool {
        let ok = *budget >= n;
        *budget = budget.saturating_sub(n);
        ok
    };
    // A folder the read guard refuses (`folo mcp --confine`) is not looked into (review round 3).
    if folo_data::guard::check(dir).is_err() {
        return false;
    }
    // The names detection takes without reading the folder: a YAML, YOLO folders, a Roboflow COCO file.
    let named = ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"].map(|n| dir.join(n));
    let splits = ["train", "valid", "val", "test"];
    if !charge(named.len() + 2 + 2 * splits.len()) {
        return false;
    }
    if named.iter().any(|p| p.is_file())
        || (dir.join("images").is_dir() && dir.join("labels").is_dir())
        || splits.iter().any(|s| dir.join(s).join("images").is_dir())
        || splits.iter().any(|s| dir.join(s).join("_annotations.coco.json").is_file())
    {
        return true;
    }
    let ext = |p: &Path, want: &[&str]| {
        p.extension().and_then(|e| e.to_str()).is_some_and(|e| want.iter().any(|w| e.eq_ignore_ascii_case(w)))
    };
    let name = |p: &Path| p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_ascii_lowercase();
    // COCO: a JSON in `annotations/` (not captions or keypoints), or one named like COCO annotations at the root.
    let annotations = dir.join("annotations");
    let listed =
        folo_data::guard::check(&annotations).ok().and_then(|()| std::fs::read_dir(&annotations).ok());
    for e in listed.into_iter().flatten() {
        if !charge(1) {
            return false;
        }
        let Ok(e) = e else { continue };
        let (p, n) = (e.path(), name(&e.path()));
        if ext(&p, &["json"])
            && !n.starts_with("captions")
            && !n.starts_with("person_keypoints")
            && p.is_file()
        {
            return true;
        }
    }
    for e in std::fs::read_dir(dir).into_iter().flatten() {
        if !charge(1) {
            return false;
        }
        let Ok(e) = e else { continue };
        let (p, n) = (e.path(), name(&e.path()));
        let coco = ext(&p, &["json"])
            && (n.contains("coco") || n.starts_with("instances") || n.contains("annotations"));
        // YOLO: any YAML with a `train` key.
        if (coco && p.is_file()) || (ext(&p, &["yaml", "yml"]) && yaml_has_train(&p)) {
            return true;
        }
    }
    false
}

/// Whether the YAML `p` has a `train:` line in its first 64 KiB. Only a regular file the read guard allows is
/// opened (`folo mcp --confine`: a link to a file outside is not read), and on Unix without waiting for a writer, so
/// a FIFO named `*.yaml`, also one swapped in after the check, never holds registration up (review round 3).
fn yaml_has_train(p: &Path) -> bool {
    use std::io::Read;
    let regular = |m: std::io::Result<std::fs::Metadata>| m.is_ok_and(|m| m.is_file());
    if !regular(std::fs::metadata(p)) || folo_data::guard::check(p).is_err() {
        return false;
    }
    let Ok(file) = open_without_waiting(p) else { return false };
    let mut head = String::new();
    regular(file.metadata())
        && file.take(64 << 10).read_to_string(&mut head).is_ok()
        && head.lines().any(|l| l.trim_start_matches('\u{feff}').starts_with("train:"))
}

/// `O_NONBLOCK` of this Unix target (std does not name it and the server takes no `libc` dependency for one flag):
/// None on a target whose value is not known here, where the probe opens nothing rather than a file that may wait
/// (review round 4).
#[cfg(unix)]
const O_NONBLOCK: Option<i32> = if cfg!(any(
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
)) {
    Some(0x4)
} else if cfg!(any(target_os = "solaris", target_os = "illumos")) {
    Some(0x80)
} else if cfg!(any(target_os = "linux", target_os = "android")) {
    if cfg!(any(
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "mips32r6",
        target_arch = "mips64r6"
    )) {
        Some(0x80)
    } else if cfg!(any(target_arch = "sparc", target_arch = "sparc64")) {
        Some(0x4000)
    } else {
        Some(0o4000)
    }
} else {
    None
};

/// Opens `p` to read; on Unix with `O_NONBLOCK`, which a regular file ignores and which opens a FIFO (one swapped
/// in after a check) at once instead of waiting for a writer. On a Unix target whose `O_NONBLOCK` is not known it
/// opens nothing (`Unsupported`): the probe then reads no YAML, which only makes its error say less.
fn open_without_waiting(p: &Path) -> std::io::Result<std::fs::File> {
    let mut open = std::fs::OpenOptions::new();
    open.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let Some(flag) = O_NONBLOCK else {
            return Err(std::io::Error::new(std::io::ErrorKind::Unsupported, "O_NONBLOCK is not known here"));
        };
        open.custom_flags(flag);
    }
    open.open(p)
}

/// Image files under `dir` (a few levels deep) among the first `budget` entries walked, for a folder that is not a
/// dataset.
fn images_under(dir: &Path, budget: usize) -> usize {
    walkdir::WalkDir::new(dir)
        .max_depth(4)
        .into_iter()
        .take(budget)
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && folo_data::is_image_path(e.path()))
        .count()
}

/// Says that `path` is not there (`404 path_not_found`) or not a dataset (`400 unknown_format`) without walking it.
/// [`Index::build`] checks it first; registration checks it before it takes the file lease, so a folder that is not
/// a dataset never holds up a dataset deletion.
pub fn preflight(path: &str, format: DatasetFormat) -> Result<()> {
    let p = absolute(Path::new(path));
    if !p.exists() {
        return Err(data_error(
            folo_data::DataError::Io { path: p.clone(), source: std::io::ErrorKind::NotFound.into() },
            path,
        ));
    }
    folo_data::detect_format(&p, data_format(format)).map(|_| ()).map_err(|e| data_error(e, path))
}

/// The path of a dataset to register, as sent: surrounding whitespace and one pair of matching quotes removed
/// (Windows' "Copy as path" adds them). It must be absolute: a relative one would resolve against the server's
/// working directory, a folder the person who sent it never sees (`400 invalid_body`).
pub fn requested_path(raw: &str) -> Result<String> {
    let t = raw.trim();
    let t = ['"', '\''].iter().find_map(|&q| t.strip_prefix(q)?.strip_suffix(q)).unwrap_or(t).trim();
    let bad = |m: String| Error::new(StatusCode::BAD_REQUEST, "invalid_body", m);
    if t.is_empty() {
        return Err(bad("path must not be empty".into()));
    }
    if !Path::new(t).is_absolute() {
        return Err(bad(format!("'{t}' is not a full path")).hint(
            "use the folder's full path on the machine running folo serve, such as D:/data/my-dataset",
        ));
    }
    Ok(t.to_string())
}

/// The name of a dataset registered without one: its folder's name (the folder above `annotations/` for a COCO file,
/// the folder of a `data.yaml`, a file's stem otherwise), numbered `-2`, `-3`, ... while `taken` says a dataset has it.
pub fn default_name(path: &Path, taken: impl Fn(&str) -> bool) -> String {
    let name = |p: Option<&Path>| p.and_then(Path::file_name).map(|n| n.to_string_lossy().into_owned());
    let base = if path.is_file() {
        let dir = path.parent();
        let yaml =
            path.extension().is_some_and(|e| e.eq_ignore_ascii_case("yaml") || e.eq_ignore_ascii_case("yml"));
        match name(dir) {
            Some(d) if yaml => Some(d),
            Some(d) if d.eq_ignore_ascii_case("annotations") => name(dir.and_then(Path::parent)),
            _ => path.file_stem().map(|n| n.to_string_lossy().into_owned()),
        }
    } else {
        name(Some(path))
    };
    free_name(base.filter(|b| !b.trim().is_empty()).unwrap_or_else(|| "dataset".into()), taken)
}

/// `base`, or `base-2`, `base-3`, ... the first one `taken` does not claim.
pub fn free_name(base: String, taken: impl Fn(&str) -> bool) -> String {
    if !taken(&base) {
        return base;
    }
    (2..).map(|n| format!("{base}-{n}")).find(|n| !taken(n)).expect("some number is free")
}

use folo_data::paths::relative_to;
pub use folo_data::paths::{lexical, normalize as absolute};

/// Normalizes `p` without breaking symlink-before-`..` resolution on Unix: when `p` contains no `..`, returns
/// `lexical(p)` untouched so symlinked dataset bases deduplicate cleanly; when `p` contains `..`, resolves existing
/// prefixes through symlinks before popping `..` on Unix so a symlinked `data.yaml` folder with `path: ../images/birds`
/// resolves to the link target's sibling `images/birds` / `labels/birds`.
fn resolve_dotdot(p: &Path) -> PathBuf {
    if !p.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return lexical(p);
    }
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                #[cfg(unix)]
                {
                    if let Ok(canon) = std::fs::canonicalize(&out) {
                        out = canon;
                    }
                }
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// The paths whose files make up a dataset: `base`, the dataset root and everything folo-data read it from
/// ([`folo_data::LoadedDataset::sources`]: split folders and lists outside `base`, folders of every candidate image
/// and label, paths that do not exist yet), minimised so no path is inside another. Sorted.
fn watch_dirs(base: &Path, ds: &folo_data::LoadedDataset) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = vec![base.to_path_buf()];
    let mut add = |d: PathBuf| {
        if !dirs.iter().any(|w| d.starts_with(w)) {
            dirs.retain(|w| !w.starts_with(&d));
            dirs.push(d);
        }
    };
    add(resolve_dotdot(&ds.root));
    for s in &ds.sources {
        add(resolve_dotdot(s));
    }
    dirs.sort();
    dirs
}

/// The paths the dataset at `path` reads its images and labels from (its own folder, its root and every source:
/// split folders and lists, the folders of its images and labels), each resolved through links before any is
/// dropped for lying inside another, so a folder that is a junction or symlink shows where it leads. Found without
/// reading the images. An error when they cannot all be known: the description cannot be read, or one of them
/// cannot be resolved or (a folder) listed. A link to a single file is not followed beyond the folder holding it.
pub fn reads(path: &str, format: DatasetFormat) -> std::result::Result<Vec<PathBuf>, String> {
    let p = absolute(Path::new(path));
    let base = if p.is_file() { p.parent().unwrap_or(&p).to_path_buf() } else { p.clone() };
    let opts = folo_data::OpenOptions { format: data_format(format), verify_images: false };
    let ds = folo_data::open_with(&p, &opts).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for s in [base, ds.root].into_iter().chain(ds.sources) {
        let s = resolve_dotdot(&s);
        out.push(resolved(&s).map_err(|e| format!("{}: {e}", folo_data::paths::shown(&s)))?);
    }
    out.sort();
    out.dedup();
    Ok(out)
}

/// `p` canonical (links followed); a path that does not exist yet is its nearest existing parent's canonical form
/// with the rest appended. A folder must be listable.
fn resolved(p: &Path) -> std::io::Result<PathBuf> {
    let mut at = p;
    let mut rest = Vec::new();
    loop {
        match std::fs::canonicalize(at) {
            Ok(c) => {
                if rest.is_empty() && c.is_dir() {
                    std::fs::read_dir(&c)?;
                }
                return Ok(rest.iter().rev().fold(c, |c, r| c.join(r)));
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => match (at.file_name(), at.parent()) {
                (Some(name), Some(parent)) => {
                    rest.push(name);
                    at = parent;
                }
                _ => return Err(e),
            },
            Err(e) => return Err(e),
        }
    }
}

impl Index {
    /// Reads the dataset at `path` (a folder, `data.yaml` or COCO `.json`) with folo-data. Image headers are read
    /// in parallel; corrupt and missing images and images with broken labels are left out.
    /// `path` is made absolute first (see [`absolute`]), so image paths never depend on the working directory.
    /// `exclude` is passed to [`Fingerprint::of`].
    pub fn build(path: &str, format: DatasetFormat, exclude: &[PathBuf]) -> Result<Index> {
        let excluded: Vec<PathBuf> = exclude.iter().map(|x| absolute(x)).collect();
        let p = absolute(Path::new(path));
        // A folder that is not a dataset is said as that before anything walks it: the fingerprint below stats every
        // file under `base`, which takes minutes for a drive or a system folder.
        preflight(path, format)?;
        let base = if p.is_file() { p.parent().unwrap_or(&p).to_path_buf() } else { p.clone() };
        let opts = folo_data::OpenOptions { format: data_format(format), verify_images: true };
        // Fingerprint the folders *before* reading them, so a file changed while folo-data reads makes the next
        // check rebuild instead of being missed. Which folders to watch is only known after reading (split sources
        // and image lists may point outside `base`), so read again until the watched set is stable (once for a
        // self-contained dataset, twice for one with external sources).
        let mut watched = vec![base.clone()];
        let mut attempt = 0;
        // Image headers only: registration already reads every file once (for the content hash of duplicates).
        let check_opts = folo_data::check::CheckOptions { format: data_format(format), decode: false };
        let (ds, report, fingerprint) = loop {
            let fingerprint = Fingerprint::of(&watched, &excluded);
            let ds = folo_data::open_with(&p, &opts).map_err(|e| data_error(e, path))?;
            let now = watch_dirs(&base, &ds);
            attempt += 1;
            if now == watched || attempt == 3 {
                // Only run the header-check pass once the watched set has stabilized so external-source datasets
                // do not scan every image header twice.
                let report = folo_data::check::check(&p, &check_opts).map_err(|e| data_error(e, path))?;
                watched = now;
                break (ds, report, fingerprint);
            }
            watched = now;
        };
        let root = resolve_dotdot(&std::path::absolute(&ds.root).unwrap_or_else(|_| ds.root.clone()));
        let issues = report
            .issues
            .iter()
            .map(|i| DatasetIssue {
                kind: i.kind.as_str().to_string(),
                severity: match i.severity() {
                    folo_data::check::Severity::Error => IssueSeverity::Error,
                    folo_data::check::Severity::Warning => IssueSeverity::Warning,
                    folo_data::check::Severity::Info => IssueSeverity::Info,
                },
                file: relative_to(
                    &resolve_dotdot(&std::path::absolute(&i.file).unwrap_or_else(|_| i.file.clone())),
                    &root,
                ),
                line: i.line.map(|l| l.min(u32::MAX as usize) as u32),
                message: i.message.clone(),
                fix: i.fix.clone(),
            })
            .collect();
        if ds.classes.len() > MAX_CLASSES {
            return Err(Error::new(
                StatusCode::BAD_REQUEST,
                "invalid_dataset",
                format!("the dataset has {} classes; at most {MAX_CLASSES} are supported", ds.classes.len()),
            ));
        }
        let stats = api_stats(ds.stats());
        let [train, val, test] = ds.split_sizes();
        let mut images = Vec::with_capacity(train + val + test);
        for split in folo_data::Split::ALL {
            for e in ds.split(split).entries() {
                let (w, h) = match e.size {
                    Some(s) => s,
                    None => folo_data::probe_size(&e.image).unwrap_or((0, 0)),
                };
                let img = resolve_dotdot(&std::path::absolute(&e.image).unwrap_or_else(|_| e.image.clone()));
                let file = relative_to(&img, &root);
                let boxes = e
                    .boxes
                    .iter()
                    .map(|b| {
                        let [x1, y1, x2, y2] = b.clipped().pixel_xyxy(w, h);
                        BBox { class: b.class, x1, y1, x2, y2 }
                    })
                    .collect();
                images.push(IndexImage { file, split: api_split(split), width: w, height: h, boxes });
            }
        }
        let format = match ds.format {
            folo_data::Format::Coco => DatasetFormat::Coco,
            _ => DatasetFormat::Yolo,
        };
        let splits = Splits { train: train as u64, val: val as u64, test: test as u64 };
        Ok(Index {
            version: INDEX_VERSION,
            fingerprint,
            watched,
            excluded,
            base,
            root,
            format,
            classes: ds.classes,
            splits,
            stats,
            images,
            issues,
        })
    }

    /// A page of issues.
    pub fn list_issues(&self, q: &IssueQuery) -> folo_api::IssuePage {
        q.page(self.issues.iter())
    }

    /// True when the files under the watched folders are the ones the index was built from.
    pub fn is_fresh(&self) -> bool {
        self.version == INDEX_VERSION && Fingerprint::of(&self.watched, &self.excluded) == self.fingerprint
    }

    fn at(&self, image_id: &str) -> Result<&IndexImage> {
        image_id
            .parse::<usize>()
            .ok()
            .filter(|i| i.to_string() == image_id)
            .and_then(|i| self.images.get(i))
            .ok_or_else(|| Error::not_found("image", image_id))
    }

    /// The file of an image.
    pub fn image_path(&self, image_id: &str) -> Result<PathBuf> {
        let im = self.at(image_id)?;
        Ok(if Path::new(&im.file).is_absolute() { PathBuf::from(&im.file) } else { self.root.join(&im.file) })
    }

    /// Contiguous `(base_id, slice)` for `split` when `self.images` matches `self.splits` in `[train, val, test]`
    /// order (always true for [`Index::build`]; returns `None` only for hand-built or inconsistent indexes so
    /// [`Index::list_images`] falls back safely without panicking).
    fn split_slice(&self, split: Split) -> Option<(usize, &[IndexImage])> {
        let train = usize::try_from(self.splits.train).ok()?;
        let val = usize::try_from(self.splits.val).ok()?;
        let test = usize::try_from(self.splits.test).ok()?;
        if train.checked_add(val)?.checked_add(test)? != self.images.len() {
            return None;
        }
        let (start, len) = match split {
            Split::Train => (0, train),
            Split::Val => (train, val),
            Split::Test => (train + val, test),
        };
        let slice = self.images.get(start..start + len)?;
        if slice.first().is_some_and(|im| im.split != split)
            || slice.last().is_some_and(|im| im.split != split)
        {
            return None;
        }
        Some((start, slice))
    }

    /// A page of images: O(limit) via contiguous split slicing when no class filter is set, and a single pass
    /// over the requested split (or all images) when filtering by class.
    pub fn list_images(&self, q: &ImageQuery) -> ImagePage {
        let to_info = |(i, im): (usize, &IndexImage)| ImageInfo {
            id: i.to_string(),
            file: im.file.clone(),
            split: im.split,
            width: im.width,
            height: im.height,
            boxes: im.boxes.clone(),
        };
        let sliced = match q.split {
            None => Some((0usize, self.images.as_slice())),
            Some(split) => self.split_slice(split),
        };
        if let Some((base, slice)) = sliced {
            let Some(class) = q.class else {
                let total = slice.len() as u64;
                let start = usize::try_from(q.offset).unwrap_or(usize::MAX).min(slice.len());
                let limit = usize::try_from(q.limit).unwrap_or(usize::MAX);
                let end = start.saturating_add(limit).min(slice.len());
                let images = slice[start..end]
                    .iter()
                    .enumerate()
                    .map(|(j, im)| to_info((base + start + j, im)))
                    .collect();
                return ImagePage { total, images };
            };
            let mut total = 0u64;
            let mut images = Vec::new();
            for (i, im) in slice.iter().enumerate() {
                if im.boxes.iter().any(|b| b.class == class) {
                    if total >= q.offset && (images.len() as u64) < q.limit {
                        images.push(to_info((base + i, im)));
                    }
                    total += 1;
                }
            }
            return ImagePage { total, images };
        }
        // Fallback when `self.images` is not laid out contiguously by `self.splits`.
        let mut total = 0u64;
        let mut images = Vec::new();
        for (i, im) in self.images.iter().enumerate() {
            if q.split.is_none_or(|s| s == im.split)
                && q.class.is_none_or(|c| im.boxes.iter().any(|b| b.class == c))
            {
                if total >= q.offset && (images.len() as u64) < q.limit {
                    images.push(to_info((i, im)));
                }
                total += 1;
            }
        }
        ImagePage { total, images }
    }

    /// Size and ground truth of an image.
    pub fn truth(&self, image_id: &str) -> Result<(u32, u32, Vec<BBox>)> {
        let im = self.at(image_id)?;
        Ok((im.width, im.height, im.boxes.clone()))
    }
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "png" => "image/png",
        "bmp" => "image/bmp",
        "webp" => "image/webp",
        _ => "image/jpeg",
    }
}

/// The bytes of an image file, or a JPEG thumbnail whose long side is `thumb` (never upscaled; EXIF orientation
/// applied, so boxes line up). With `cache`, thumbnails are kept there as `<key>.jpg`, keyed by the file's path,
/// size and modification time and the side.
pub fn read_image(path: &Path, thumb: Option<u32>, cache: Option<&Path>) -> Result<ImageData> {
    read_image_with(path, thumb, cache, |file, bytes| {
        write_atomic(file, bytes)
            .unwrap_or_else(|e| tracing::warn!("caching thumbnail {}: {e}", file.display()))
    })
}

fn missing_image_error(path: &Path, e: std::io::Error) -> Error {
    if e.kind() == std::io::ErrorKind::NotFound {
        Error::new(StatusCode::NOT_FOUND, "image_not_found", format!("{} no longer exists", path.display()))
            .hint("the dataset changed on disk; reload the image list")
    } else {
        Error::internal(format!("reading {}: {e}", path.display()))
    }
}

/// Reads and decodes a dataset image file on disk (`404 image_not_found` when the file disappeared since indexing,
/// `422 image_decode_failed` when its bytes cannot be decoded).
pub fn decode_image(path: &Path) -> Result<folo_data::Image> {
    folo_data::read_image(path).map_err(|e| match e {
        folo_data::DataError::Io { source, .. } => missing_image_error(path, source),
        e => Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string()),
    })
}

/// [`read_image`] that hands a newly made thumbnail to `save(cache_file, bytes)` instead of writing it, so the
/// caller can coordinate cache writes (e.g. with unregistering the dataset).
pub fn read_image_with(
    path: &Path,
    thumb: Option<u32>,
    cache: Option<&Path>,
    save: impl FnOnce(&Path, &[u8]),
) -> Result<ImageData> {
    let missing = |e: std::io::Error| missing_image_error(path, e);
    folo_data::guard::check(path).map_err(missing)?;
    let Some(side) = thumb else {
        let bytes = std::fs::read(path).map_err(missing)?;
        return Ok(ImageData { content_type: content_type(path), bytes, immutable: false });
    };
    let md = std::fs::metadata(path).map_err(missing)?;
    let mtime =
        md.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos());
    let key = format!("{}|{}|{mtime}|{side}", path.display(), md.len());
    let cached = cache.map(|dir| dir.join(format!("{:016x}.jpg", fnv1a(key.as_bytes()))));
    if let Some(c) = &cached
        && let Ok(bytes) = std::fs::read(c)
    {
        return Ok(ImageData { content_type: "image/jpeg", bytes, immutable: false });
    }
    let img = decode_image(path)?;
    let long = img.width.max(img.height);
    let img = if side < long {
        let scale = side as f64 / long as f64;
        let w = ((img.width as f64 * scale).round() as u32).max(1);
        let h = ((img.height as f64 * scale).round() as u32).max(1);
        folo_data::resize::resize(&img, w, h)
    } else {
        img
    };
    let bytes = folo_data::encode(&img, folo_data::ImageFormat::Jpeg(85))
        .map_err(|e| Error::internal(e.to_string()))?;
    if let Some(c) = &cached {
        save(c, &bytes);
    }
    Ok(ImageData { content_type: "image/jpeg", bytes, immutable: false })
}

/// Writes `bytes` to a unique temporary file next to `path`, then renames it over `path`, so readers never see a
/// partial file (concurrent writers of the same content are harmless). A `path` another program holds open is
/// waited for only [briefly](folo_nn::io::RenameRetry::BRIEF): the store's files are written under the locks of
/// the state they persist (a run's `run.json` under its run's), which a long wait would stall.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
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
    let write_and_rename = || -> std::io::Result<()> {
        std::fs::write(&tmp, bytes)?;
        folo_nn::io::rename_over_with(&tmp, path, STORE_RETRY).map(drop)
    };
    write_and_rename().inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// How [`write_atomic`] waits for a held file, per write: an operation that makes several writes under one lock (a
/// dataset registration makes three) may hold it for that many budgets.
const STORE_RETRY: folo_nn::io::RenameRetry = folo_nn::io::RenameRetry::BRIEF;

#[cfg(test)]
mod tests;
