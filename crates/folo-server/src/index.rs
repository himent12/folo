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

/// Bumped whenever the cached [`Index`] layout or its derivation changes, so old caches are rebuilt.
pub const INDEX_VERSION: u32 = 4;

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
        DataError::Config(m) => Error::new(StatusCode::BAD_REQUEST, "invalid_body", m),
        e => Error::internal(e.to_string()),
    }
}

/// Entries the diagnostics of a folder that is not a dataset look at, at most ([`dataset_below`], [`images_under`]):
/// they only phrase an error, so a drive or a system folder is never walked to say it is not one (x3-server-1).
pub(crate) const DIAGNOSE_ENTRIES: usize = 20_000;

/// The first folder one or two levels below `dir` (in name order, at most 64 folders a level, `budget` folder entries
/// read in all) where a dataset is detected, for a folder that is not a dataset itself.
pub(crate) fn dataset_below(dir: &Path, mut budget: usize) -> Option<PathBuf> {
    fn folders(dir: &Path, budget: &mut usize) -> Vec<PathBuf> {
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
    let detected = |d: &Path| folo_data::detect_format(d, folo_data::Format::Auto).is_ok();
    let first = folders(dir, &mut budget);
    if let Some(d) = first.iter().find(|d| detected(d)) {
        return Some(d.clone());
    }
    for d in &first {
        if let Some(found) = folders(d, &mut budget).into_iter().find(|d| detected(d)) {
            return Some(found);
        }
    }
    None
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
    add(lexical(&ds.root));
    for s in &ds.sources {
        add(lexical(s));
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
        let s = lexical(&s);
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
            let report = folo_data::check::check(&p, &check_opts).map_err(|e| data_error(e, path))?;
            let now = watch_dirs(&base, &ds);
            attempt += 1;
            if now == watched || attempt == 3 {
                watched = now;
                break (ds, report, fingerprint);
            }
            watched = now;
        };
        let root = absolute(&ds.root);
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
                file: relative_to(&i.file, &root),
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
                let file = relative_to(&e.image, &root);
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

    /// A page of images.
    pub fn list_images(&self, q: &ImageQuery) -> ImagePage {
        let matches = |im: &IndexImage| {
            q.split.is_none_or(|s| s == im.split)
                && q.class.is_none_or(|c| im.boxes.iter().any(|b| b.class == c))
        };
        let total = self.images.iter().filter(|im| matches(im)).count() as u64;
        let images = self
            .images
            .iter()
            .enumerate()
            .filter(|(_, im)| matches(im))
            .skip(q.offset.min(usize::MAX as u64) as usize)
            .take(q.limit.min(usize::MAX as u64) as usize)
            .map(|(i, im)| ImageInfo {
                id: i.to_string(),
                file: im.file.clone(),
                split: im.split,
                width: im.width,
                height: im.height,
                boxes: im.boxes.clone(),
            })
            .collect();
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

/// [`read_image`] that hands a newly made thumbnail to `save(cache_file, bytes)` instead of writing it, so the
/// caller can coordinate cache writes (e.g. with unregistering the dataset).
pub fn read_image_with(
    path: &Path,
    thumb: Option<u32>,
    cache: Option<&Path>,
    save: impl FnOnce(&Path, &[u8]),
) -> Result<ImageData> {
    let missing = |e: std::io::Error| {
        if e.kind() == std::io::ErrorKind::NotFound {
            Error::new(
                StatusCode::NOT_FOUND,
                "image_not_found",
                format!("{} no longer exists", path.display()),
            )
            .hint("the dataset changed on disk; reload the image list")
        } else {
            Error::internal(format!("reading {}: {e}", path.display()))
        }
    };
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
    let img = folo_data::read_image(path).map_err(|e| match e {
        folo_data::DataError::Io { source, .. } => missing(source),
        e => Error::new(StatusCode::UNPROCESSABLE_ENTITY, "image_decode_failed", e.to_string()),
    })?;
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
/// partial file (concurrent writers of the same content are harmless).
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
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use folo_test_support::TestDir;

    fn tmp(tag: &str) -> TestDir {
        TestDir::new(&format!("index-{tag}"))
    }

    #[test]
    fn default_names_come_from_the_folder_and_are_numbered_when_taken() {
        let dir = tmp("names");
        let bccd = dir.join("bccd");
        std::fs::create_dir_all(bccd.join("annotations")).unwrap();
        for f in ["annotations/instances_test.json", "data.yaml", "loose.json"] {
            std::fs::write(bccd.join(f), "{}").unwrap();
        }
        let free = |_: &str| false;
        assert_eq!(default_name(&bccd, free), "bccd");
        assert_eq!(default_name(&bccd.join("annotations/instances_test.json"), free), "bccd");
        assert_eq!(default_name(&bccd.join("data.yaml"), free), "bccd");
        assert_eq!(default_name(&bccd.join("loose.json"), free), "loose");
        let taken = |n: &str| ["bccd", "bccd-2"].contains(&n.to_lowercase().as_str());
        assert_eq!(default_name(&bccd, taken), "bccd-3");
    }

    /// The diagnostics of a folder that is not a dataset stop after their budget of entries, whatever they find
    /// (x3-server-1: they used to count up to 10,000 *images*, so a wide tree without images was walked whole).
    #[test]
    fn diagnostics_of_a_non_dataset_read_a_bounded_number_of_entries() {
        let dir = tmp("diagnose");
        for i in 0..40 {
            std::fs::create_dir_all(dir.join(format!("sub{i:02}"))).unwrap();
            std::fs::write(dir.join(format!("{i:02}.jpg")), b"x").unwrap();
        }
        std::fs::create_dir_all(dir.join("sub39/set/images")).unwrap();
        std::fs::write(dir.join("sub39/set/data.yaml"), "names: [x]\n").unwrap();
        assert_eq!(images_under(&dir, 1000), 40);
        // The folder itself is the first entry walked: at most 9 of the 40 images are seen.
        assert!(images_under(&dir, 10) < 10, "the walk stops after 10 entries");
        assert!(dataset_below(&dir, 1000).is_some());
        assert_eq!(dataset_below(&dir, 10), None, "the folder search stops after 10 entries");
    }

    #[test]
    fn requested_paths_lose_quotes_and_must_be_absolute() {
        let dir = tmp("requested");
        let p = dir.to_str().unwrap();
        for raw in [
            p.to_string(),
            format!("\"{p}\""),
            format!("  '{p}' "),
            format!(
                "\" {p}\"
"
            ),
        ] {
            assert_eq!(requested_path(&raw).unwrap(), p, "{raw:?}");
        }
        for raw in ["bccd", "\"data/x\"", "  ", "\"\""] {
            let e = requested_path(raw).unwrap_err();
            assert_eq!(
                (e.status, e.body.code.as_str()),
                (StatusCode::BAD_REQUEST, "invalid_body"),
                "{raw:?}"
            );
        }
        assert_eq!(requested_path("bccd").unwrap_err().body.message, "'bccd' is not a full path");
    }

    #[test]
    fn build_list_and_fingerprint() {
        let dir = tmp("build");
        for s in ["train", "val"] {
            std::fs::create_dir_all(dir.join("images").join(s)).unwrap();
            std::fs::create_dir_all(dir.join("labels").join(s)).unwrap();
        }
        std::fs::write(dir.join("data.yaml"), "train: images/train\nval: images/val\nnames: [cat, dog]\n")
            .unwrap();
        let img = image::RgbImage::from_pixel(40, 20, image::Rgb([200, 10, 10]));
        img.save(dir.join("images/train/a.png")).unwrap();
        img.save(dir.join("images/val/b.png")).unwrap();
        std::fs::write(dir.join("labels/train/a.txt"), "1 0.5 0.5 0.5 0.5\n0 0.995 0.5 0.02 0.2\n").unwrap();
        let idx = Index::build(dir.to_str().unwrap(), DatasetFormat::Auto, &[]).unwrap();
        assert_eq!(idx.classes, vec!["cat", "dog"]);
        assert_eq!(idx.splits, Splits { train: 1, val: 1, test: 0 });
        assert_eq!(idx.stats.class_counts, vec![1, 1]);
        assert_eq!(idx.stats.empty_images, 1);
        let page = idx.list_images(&ImageQuery { limit: 10, ..Default::default() });
        assert_eq!(page.total, 2);
        assert_eq!(page.images[0].file, "images/train/a.png");
        assert_eq!((page.images[0].width, page.images[0].boxes[0].x1), (40, 10.0));
        // A box poking out of the right edge is clipped, never inverted.
        let edge = page.images[0].boxes[1];
        assert!(edge.x1 < edge.x2 && edge.x2 == 40.0, "{edge:?}");
        let dog = idx.list_images(&ImageQuery { limit: 10, class: Some(1), ..Default::default() });
        assert_eq!(dog.total, 1);
        assert!(idx.is_fresh());
        assert!(idx.at("01").is_err() && idx.at("2").is_err());

        let cache = dir.join("cache");
        let path = idx.image_path("0").unwrap();
        let th = read_image(&path, Some(10), Some(&cache)).unwrap();
        let decoded = image::load_from_memory(&th.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (10, 5));
        assert_eq!(std::fs::read_dir(&cache).unwrap().count(), 1, "thumbnail cached");
        assert_eq!(read_image(&path, Some(10), Some(&cache)).unwrap().bytes, th.bytes);
        assert!(!read_image(&path, Some(4096), None).unwrap().bytes.is_empty());

        std::fs::write(dir.join("labels/train/a.txt"), "1 0.5 0.5 0.5 0.5\n").unwrap();
        assert!(!idx.is_fresh(), "editing a label invalidates the index");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn errors_map_to_api_codes() {
        let dir = tmp("errors");
        let code =
            |p: &Path| Index::build(p.to_str().unwrap(), DatasetFormat::Auto, &[]).unwrap_err().body.code;
        assert_eq!(code(&dir.join("nope")), "path_not_found");
        assert_eq!(code(&dir), "unknown_format");
        std::fs::write(dir.join("data.yaml"), "train: images\nnc: 2\nnames: [a]\n").unwrap();
        assert_eq!(code(&dir), "invalid_dataset");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
