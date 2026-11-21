//! Datasets: the [`Dataset`] trait, YOLO and COCO readers with format auto-detection, and the load report.
//!
//! - `yolo.rs`: the YOLO reader (`data.yaml`, `images/` + `labels/`) and [`label_path_for`]; `labels.rs`: YOLO
//!   label text parsing (boxes and segment polygons).
//! - `coco.rs`: COCO instances JSON, and the ground-truth reader [`open_coco_ground_truth`].
//! - `cache/`: the on-disk scan cache (docs/DATA.md, Scan cache).
//! - This file: [`Dataset`], [`FileDataset`], [`InMemoryDataset`], [`Split`], [`Format`], [`LoadReport`],
//!   [`open`] / [`open_with`] / [`open_splits`] and [`detect_format`].

pub(crate) mod cache;
mod coco;

pub use coco::{
    CocoAnnotation, CocoGroundTruth, CocoImage, coco_annotation_files, open_coco_ground_truth,
    read_coco_ground_truth,
};
pub(crate) mod labels;
pub(crate) mod yolo;

pub use yolo::label_path_for;

use crate::bbox::BBox;
use crate::error::{DataError, Result};
use crate::image::{Sample, probe_size, read_image};
use crate::stats::DatasetStats;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A dataset split.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Split {
    /// Training images.
    Train,
    /// Validation images.
    Val,
    /// Test images.
    Test,
}

impl Split {
    /// All splits in order.
    pub const ALL: [Split; 3] = [Split::Train, Split::Val, Split::Test];

    /// Lower-case name (`train`, `val`, `test`).
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Val => "val",
            Split::Test => "test",
        }
    }
}

impl std::fmt::Display for Split {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// On-disk dataset format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// Detect from the files present.
    #[default]
    Auto,
    /// YOLO txt labels (`images/` + `labels/`, optional `data.yaml`).
    Yolo,
    /// COCO instances JSON.
    Coco,
}

/// Random access to labelled images.
///
/// Labels are kept in memory (so [`Dataset::boxes`] and statistics never touch the disk); images are decoded on
/// [`Dataset::get`]. Implementations must be thread-safe: the [`crate::DataLoader`] calls `get` from many threads.
pub trait Dataset: Send + Sync {
    /// Number of samples.
    fn len(&self) -> usize;

    /// Whether the dataset has no samples.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Decodes sample `index`. Fails for unreadable images; never panics for bad data.
    fn get(&self, index: usize) -> Result<Sample>;

    /// The boxes of sample `index` without decoding the image.
    fn boxes(&self, index: usize) -> &[BBox];

    /// Class names; box `class` ids index into this.
    fn class_names(&self) -> &[String];

    /// Which split this is.
    fn split(&self) -> Split;

    /// Image `(width, height)` if known without decoding.
    fn image_size(&self, _index: usize) -> Option<(u32, u32)> {
        None
    }

    /// Image file path, for file-backed datasets.
    fn image_path(&self, _index: usize) -> Option<&Path> {
        None
    }

    /// Statistics over all samples (labels only; no image decoding).
    fn stats(&self) -> DatasetStats {
        DatasetStats::compute(self.class_names().len(), (0..self.len()).map(|i| self.boxes(i)), 0)
    }
}

/// One image of a file-backed dataset.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageEntry {
    /// Image file.
    pub image: PathBuf,
    /// YOLO label file, when there is one on disk.
    pub label: Option<PathBuf>,
    /// `(width, height)` after EXIF orientation, when known.
    pub size: Option<(u32, u32)>,
    /// Boxes in normalised `cx, cy, w, h`.
    pub boxes: Vec<BBox>,
}

/// A split of an on-disk dataset (YOLO or COCO).
#[derive(Clone, Debug)]
pub struct FileDataset {
    split: Split,
    classes: Vec<String>,
    entries: Vec<ImageEntry>,
}

impl FileDataset {
    /// Builds a dataset from entries.
    pub fn new(split: Split, classes: Vec<String>, entries: Vec<ImageEntry>) -> Self {
        FileDataset { split, classes, entries }
    }

    /// The images and their labels.
    pub fn entries(&self) -> &[ImageEntry] {
        &self.entries
    }
}

impl Dataset for FileDataset {
    fn len(&self) -> usize {
        self.entries.len()
    }

    fn get(&self, index: usize) -> Result<Sample> {
        let e = self.entries.get(index).ok_or(DataError::OutOfRange { index, len: self.entries.len() })?;
        let image = read_image(&e.image)?;
        Ok(Sample { image, boxes: e.boxes.clone() })
    }

    fn boxes(&self, index: usize) -> &[BBox] {
        &self.entries[index].boxes
    }

    fn class_names(&self) -> &[String] {
        &self.classes
    }

    fn split(&self) -> Split {
        self.split
    }

    fn image_size(&self, index: usize) -> Option<(u32, u32)> {
        self.entries.get(index).and_then(|e| e.size)
    }

    fn image_path(&self, index: usize) -> Option<&Path> {
        self.entries.get(index).map(|e| e.image.as_path())
    }
}

/// A dataset held entirely in memory.
#[derive(Clone, Debug)]
pub struct InMemoryDataset {
    /// The samples.
    pub samples: Vec<Sample>,
    /// Class names.
    pub classes: Vec<String>,
    /// Split tag.
    pub split: Split,
}

impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Result<Sample> {
        self.samples.get(index).cloned().ok_or(DataError::OutOfRange { index, len: self.samples.len() })
    }

    fn boxes(&self, index: usize) -> &[BBox] {
        &self.samples[index].boxes
    }

    fn class_names(&self) -> &[String] {
        &self.classes
    }

    fn split(&self) -> Split {
        self.split
    }

    fn image_size(&self, index: usize) -> Option<(u32, u32)> {
        self.samples.get(index).map(|s| (s.image.width, s.image.height))
    }
}

/// What kind of problem an [`Issue`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IssueKind {
    /// The image file exists but cannot be decoded; the image was skipped.
    CorruptImage,
    /// The image file is missing; it was skipped.
    MissingImage,
    /// The label file (or COCO annotation) is malformed; the image was skipped.
    BrokenLabel,
    /// A fixable label problem (duplicate, clipped or zero-size box); the image was kept.
    LabelWarning,
    /// Anything else worth knowing (a missing split, a size mismatch, ...).
    Other,
}

/// One problem found while loading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Issue {
    /// The kind of problem.
    pub kind: IssueKind,
    /// The file involved.
    pub path: PathBuf,
    /// Human-readable details.
    pub message: String,
}

/// Problems found while opening a dataset. Nothing in it is fatal: bad images and labels are skipped.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadReport {
    /// All issues, in discovery order.
    pub issues: Vec<Issue>,
}

impl LoadReport {
    /// Number of issues of `kind`.
    pub fn count(&self, kind: IssueKind) -> usize {
        self.issues.iter().filter(|i| i.kind == kind).count()
    }

    /// Broken labels ([`IssueKind::BrokenLabel`] issues): a YOLO label file, or a COCO image's annotations, that
    /// cannot be read (one however many broken lines or annotations), whose image is skipped, and for COCO also a
    /// category, image or annotation entry or a whole annotation file that cannot be read, dropped without skipping
    /// the other images. Not a count of images.
    pub fn broken_labels(&self) -> usize {
        self.count(IssueKind::BrokenLabel)
    }

    /// Images skipped because they are missing or cannot be decoded.
    pub fn skipped_images(&self) -> usize {
        self.count(IssueKind::CorruptImage) + self.count(IssueKind::MissingImage)
    }

    pub(crate) fn push(&mut self, kind: IssueKind, path: impl Into<PathBuf>, message: impl Into<String>) {
        self.issues.push(Issue { kind, path: path.into(), message: message.into() });
    }
}

/// Options for [`open_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpenOptions {
    /// Format to read; `Auto` detects it.
    pub format: Format,
    /// Read every image header while opening (in parallel) so corrupt files are reported and skipped up front and
    /// image sizes are known. Default `true`.
    pub verify_images: bool,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions { format: Format::Auto, verify_images: true }
    }
}

/// An opened on-disk dataset: its splits, class names and load report.
#[derive(Clone, Debug)]
pub struct LoadedDataset {
    /// The format that was read (never `Auto`).
    pub format: Format,
    /// Dataset root directory.
    pub root: PathBuf,
    /// Class names.
    pub classes: Vec<String>,
    /// Training split.
    pub train: FileDataset,
    /// Validation split.
    pub val: FileDataset,
    /// Test split (often empty).
    pub test: FileDataset,
    /// Problems found while loading.
    pub report: LoadReport,
    /// Every file and folder the dataset was read from, for change detection: the `data.yaml` or COCO JSON files,
    /// split folders, image lists and images named in the YAML (also ones that do not exist yet), and the folder of
    /// every candidate image and its label file (also of images that were skipped). A change under any of them may
    /// change the dataset.
    pub sources: Vec<PathBuf>,
}

impl LoadedDataset {
    /// One split.
    pub fn split(&self, split: Split) -> &FileDataset {
        match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
            Split::Test => &self.test,
        }
    }

    /// Statistics over all splits, with `broken_labels` from the load report.
    pub fn stats(&self) -> DatasetStats {
        let boxes = Split::ALL.into_iter().flat_map(|s| {
            let d = self.split(s);
            (0..d.len()).map(move |i| d.boxes(i))
        });
        DatasetStats::compute(self.classes.len(), boxes, self.report.broken_labels() as u64)
    }

    /// Number of images per split, `[train, val, test]`.
    pub fn split_sizes(&self) -> [usize; 3] {
        [self.train.len(), self.val.len(), self.test.len()]
    }
}

/// Opens a dataset at `path`, detecting the format. See [`open_with`].
pub fn open(path: impl AsRef<Path>) -> Result<LoadedDataset> {
    open_with(path, &OpenOptions::default())
}

/// Opens a dataset.
///
/// `path` may be a YOLO `data.yaml`, a COCO instances `.json`, or a directory. In a directory, a `data.yaml` (or
/// another YAML with a `train` key) means YOLO; COCO JSON under `annotations/`, the root or `train|val|test/` means
/// COCO; `images/` + `labels/` folders without a YAML mean YOLO as well.
pub fn open_with(path: impl AsRef<Path>, opts: &OpenOptions) -> Result<LoadedDataset> {
    open_splits(path, opts, &Split::ALL)
}

/// [`open_with`] for some splits only: the splits in `splits` come out exactly as [`open_with`] builds them (same
/// classes, entries and order), the others empty, and their images and labels are never read or probed (COCO
/// files of the other splits are read for their categories only). The report and [`LoadedDataset::sources`] cover
/// what was read. `folo val` opens just the split it evaluates this way: COCO's 5 000 validation images without
/// checking the 118 287 training images first.
///
/// A YOLO dataset without class names (they are then counted from the labels of every split) is opened whole.
pub fn open_splits(path: impl AsRef<Path>, opts: &OpenOptions, splits: &[Split]) -> Result<LoadedDataset> {
    let path = path.as_ref();
    match detect(path, opts.format)? {
        Source::Yaml(yaml) => yolo::finish(yolo::scan_yaml(&yaml)?, opts, splits),
        Source::Coco { root, jsons } => coco::open_jsons(&root, jsons, opts, splits),
        Source::YoloDir(root) => yolo::finish(yolo::scan_dir(&root), opts, splits),
    }
}

/// Where a dataset's description lives, as found by [`detect`].
pub(crate) enum Source {
    /// A YOLO `data.yaml`.
    Yaml(PathBuf),
    /// COCO instances JSON files and the dataset root they resolve images against.
    Coco { root: PathBuf, jsons: Vec<(Split, PathBuf)> },
    /// A YOLO `images/` + `labels/` folder without a YAML.
    YoloDir(PathBuf),
}

/// Finds the dataset at `path` (rules of [`open_with`]) without reading any labels or images. A folder a writer has
/// not finished ([`crate::unfinished`]) is refused once found: the YAML's folder (and the root its `path:` gives,
/// `yolo::scan_yaml`), the folder images resolve against, or a COCO JSON's own folder or the one above it.
pub(crate) fn detect(path: &Path, format: Format) -> Result<Source> {
    let source = locate(path, format)?;
    match &source {
        Source::Yaml(yaml) => crate::unfinished::check(crate::paths::parent_dir(yaml))?,
        Source::YoloDir(root) => crate::unfinished::check(root)?,
        Source::Coco { root, jsons } => {
            crate::unfinished::check(root)?;
            // By plain path, names not resolved: a writer starts in an empty folder, so all below a marked folder is
            // its unfinished output (`annotations/x.json` and `ANNOTATIONS/x.json` alike). The folder above is
            // `dir/..`, which the file system resolves (`partial/../x.json` is beside `partial`, not in it).
            for (_, json) in jsons {
                let dir = crate::paths::parent_dir(json);
                crate::unfinished::check(dir)?;
                crate::unfinished::check(&dir.join(".."))?;
            }
        }
    }
    Ok(source)
}

fn locate(path: &Path, format: Format) -> Result<Source> {
    if !path.exists() {
        return Err(DataError::io(
            path,
            std::io::Error::new(std::io::ErrorKind::NotFound, "dataset path not found"),
        ));
    }
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
    if path.is_file() {
        return match (format, ext.as_deref()) {
            (Format::Auto | Format::Yolo, Some("yaml" | "yml")) => Ok(Source::Yaml(path.to_path_buf())),
            (Format::Auto | Format::Coco, Some("json")) => {
                let parent = crate::paths::parent_dir(path);
                let root =
                    if parent.ends_with("annotations") { crate::paths::parent_dir(parent) } else { parent };
                Ok(Source::Coco {
                    root: root.to_path_buf(),
                    jsons: vec![(coco::guess_split(path), path.to_path_buf())],
                })
            }
            // Said in the words people pick the format with ("YOLO", "COCO"), not the enum's.
            (Format::Yolo, Some("json")) => Err(DataError::Format(format!(
                "{} is a COCO .json file, not YOLO: read it as COCO (or detect the format)",
                crate::paths::shown(path),
            ))),
            (Format::Coco, Some("yaml" | "yml")) => Err(DataError::Format(format!(
                "{} is a YOLO data.yaml, not COCO: read it as YOLO (or detect the format)",
                crate::paths::shown(path),
            ))),
            _ => Err(DataError::Format(format!(
                "{} is not a data.yaml or a COCO .json file",
                crate::paths::shown(path),
            ))),
        };
    }
    let want_yolo = matches!(format, Format::Auto | Format::Yolo);
    let want_coco = matches!(format, Format::Auto | Format::Coco);
    if want_yolo && let Some(yaml) = yolo::find_yaml(path) {
        return Ok(Source::Yaml(yaml));
    }
    if want_coco {
        let jsons = coco::find_jsons(path);
        if !jsons.is_empty() {
            return Ok(Source::Coco { root: path.to_path_buf(), jsons });
        }
    }
    if want_yolo && yolo::looks_like_yolo_dir(path) {
        return Ok(Source::YoloDir(path.to_path_buf()));
    }
    Err(DataError::Format(format!(
        "{UNDETECTED} in {}: expected a data.yaml, COCO instances JSON, or images/ + labels/ folders",
        crate::paths::shown(path)
    )))
}

/// Start of the [`DataError::Format`] message when no dataset is found at a path (see [`detect_format`]).
pub const UNDETECTED: &str = "could not detect a dataset";

/// The format of the dataset at `path` without reading it (`format` restricts the search; `Auto` tries both).
/// Fails with [`DataError::Io`] when `path` does not exist and with [`DataError::Format`] (message starting with
/// [`UNDETECTED`]) when nothing there looks like a dataset.
pub fn detect_format(path: impl AsRef<Path>, format: Format) -> Result<Format> {
    Ok(match detect(path.as_ref(), format)? {
        Source::Yaml(_) | Source::YoloDir(_) => Format::Yolo,
        Source::Coco { .. } => Format::Coco,
    })
}

/// How labels for one candidate image are obtained.
pub(crate) enum LabelSource {
    /// A YOLO txt file (may not exist: background image).
    Yolo(PathBuf),
    /// COCO boxes in pixel `xyxy`, with the size the JSON declares, or a reason the annotations are broken.
    Coco { boxes: Vec<(u32, [f32; 4])>, declared: Option<(u32, u32)>, broken: Option<String>, label: PathBuf },
}

pub(crate) struct Candidate {
    pub image: PathBuf,
    pub labels: LabelSource,
}

/// Validates candidates in parallel (order preserved), returning kept entries and issues.
pub(crate) fn build_entries(
    candidates: Vec<Candidate>,
    num_classes: Option<usize>,
    opts: &OpenOptions,
    report: &mut LoadReport,
) -> Vec<ImageEntry> {
    let files = candidates.iter().flat_map(|c| {
        let label = match &c.labels {
            LabelSource::Yolo(l) => Some(l.as_path()),
            LabelSource::Coco { .. } => None,
        };
        std::iter::once(c.image.as_path()).chain(label)
    });
    let probes = cache::probe::Probes::new(files, cache::Settings::current());
    let results: Vec<Checked> =
        candidates.into_par_iter().map(|c| check_candidate(c, num_classes, opts, &probes)).collect();
    let mut entries = Vec::with_capacity(results.len());
    let mut fresh = Vec::new();
    for (e, issues, learnt) in results {
        report.issues.extend(issues);
        entries.extend(e);
        fresh.extend(learnt);
    }
    probes.save(fresh);
    entries
}

/// A checked candidate: its entry (unless skipped), its issues and what the scan cache should learn from it.
type Checked = (Option<ImageEntry>, Vec<Issue>, Vec<cache::probe::Fresh>);

/// The header probe of `image` (`(width, height)` or the issue), from the scan cache when it has it. A file gone
/// since it was listed is a [`IssueKind::MissingImage`], any other failure a [`IssueKind::CorruptImage`].
fn probe_image(
    image: &Path,
    stamp: Option<cache::Stamp>,
    probes: &cache::probe::Probes,
    fresh: &mut Vec<cache::probe::Fresh>,
) -> std::result::Result<(u32, u32), (IssueKind, String)> {
    use cache::probe::{Fresh, ImageProbe};
    if let Some(s) = stamp
        && let Some(p) = probes.image(image, s)
    {
        return match p {
            ImageProbe::Size(w, h) => Ok((*w, *h)),
            ImageProbe::Corrupt(m) => Err((IssueKind::CorruptImage, DataError::image(image, m).to_string())),
        };
    }
    let probed = probe_size(image);
    if let Some(s) = probes.cacheable(image, stamp) {
        // Only what the file's bytes decide is cached, not a failure to read them.
        match &probed {
            Ok((w, h)) => fresh.push(Fresh::Image(image.to_path_buf(), s, ImageProbe::Size(*w, *h))),
            Err(DataError::Image { message, .. }) => {
                fresh.push(Fresh::Image(image.to_path_buf(), s, ImageProbe::Corrupt(message.clone())))
            }
            Err(_) => {}
        }
    }
    probed.map_err(|e| match e {
        DataError::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            (IssueKind::MissingImage, "image file not found".to_string())
        }
        e => (IssueKind::CorruptImage, e.to_string()),
    })
}

/// The YOLO label file `label` read and parsed, from the scan cache when it has it. `Err` is the message of a
/// [`IssueKind::BrokenLabel`] issue.
fn parse_label(
    label: &Path,
    stamp: Option<cache::Stamp>,
    num_classes: Option<usize>,
    probes: &cache::probe::Probes,
    fresh: &mut Vec<cache::probe::Fresh>,
) -> std::result::Result<(Vec<BBox>, Vec<String>), String> {
    use cache::probe::{Fresh, LabelParse};
    if let Some(s) = stamp
        && let Some(p) = probes.label(label, s, num_classes)
    {
        return match p {
            LabelParse::Parsed { boxes, warnings } => Ok((boxes.clone(), warnings.clone())),
            LabelParse::Broken(m) => Err(m.clone()),
        };
    }
    let bytes = match crate::guard::check(label).and_then(|_| std::fs::read(label)) {
        Ok(b) => b,
        // A failed read is not cached.
        Err(e) => return Err(e.to_string()),
    };
    let parsed = match String::from_utf8(bytes) {
        Ok(text) => labels::parse_yolo(&text, num_classes)
            .map(|p| (p.boxes, p.warnings))
            .map_err(|(line, msg)| format!("line {line}: {msg}")),
        Err(_) => Err("label file is not UTF-8 text".to_string()),
    };
    if let Some(s) = probes.cacheable(label, stamp) {
        let p = match &parsed {
            Ok((boxes, warnings)) => LabelParse::Parsed { boxes: boxes.clone(), warnings: warnings.clone() },
            Err(m) => LabelParse::Broken(m.clone()),
        };
        fresh.push(Fresh::Label(label.to_path_buf(), s, num_classes, p));
    }
    parsed
}

fn check_candidate(
    c: Candidate,
    num_classes: Option<usize>,
    opts: &OpenOptions,
    probes: &cache::probe::Probes,
) -> Checked {
    let mut issues = Vec::new();
    let mut fresh = Vec::new();
    let issue = |kind, path: &Path, message: String| Issue { kind, path: path.to_path_buf(), message };
    let Some(image) = probes.stat(&c.image).filter(|e| e.is_file) else {
        issues.push(issue(IssueKind::MissingImage, &c.image, "image file not found".into()));
        return (None, issues, fresh);
    };
    let need_size = opts.verify_images || matches!(&c.labels, LabelSource::Coco { declared: None, .. });
    let size = if need_size {
        match probe_image(&c.image, image.stamp, probes, &mut fresh) {
            Ok(s) => Some(s),
            Err((kind, m)) => {
                issues.push(issue(kind, &c.image, m));
                return (None, issues, fresh);
            }
        }
    } else {
        None
    };
    match c.labels {
        LabelSource::Yolo(label) => {
            let (boxes, label) = match probes.stat(&label).filter(|e| e.is_file) {
                Some(l) => match parse_label(&label, l.stamp, num_classes, probes, &mut fresh) {
                    Ok((boxes, warnings)) => {
                        for w in warnings {
                            issues.push(issue(IssueKind::LabelWarning, &label, w));
                        }
                        (boxes, Some(label))
                    }
                    Err(m) => {
                        issues.push(issue(IssueKind::BrokenLabel, &label, m));
                        return (None, issues, fresh);
                    }
                },
                None => (Vec::new(), None),
            };
            (Some(ImageEntry { image: c.image, label, size, boxes }), issues, fresh)
        }
        LabelSource::Coco { boxes, declared, broken, label } => {
            if let Some(msg) = broken {
                issues.push(issue(
                    IssueKind::BrokenLabel,
                    &label,
                    format!("{}: {msg}", crate::paths::shown(&c.image)),
                ));
                return (None, issues, fresh);
            }
            if let (Some(d), Some(s)) = (declared, size)
                && d != s
            {
                issues.push(issue(
                    IssueKind::Other,
                    &c.image,
                    format!(
                        "JSON says {}x{} but the image is {}x{}; using the image size",
                        d.0, d.1, s.0, s.1
                    ),
                ));
            }
            let Some((w, h)) = size.or(declared) else {
                issues.push(issue(IssueKind::CorruptImage, &c.image, "unknown image size".into()));
                return (None, issues, fresh);
            };
            let (fw, fh) = (w as f32, h as f32);
            let mut out = Vec::with_capacity(boxes.len());
            let mut dropped = 0;
            for (class, [x1, y1, x2, y2]) in boxes {
                let c = [x1.clamp(0.0, fw), y1.clamp(0.0, fh), x2.clamp(0.0, fw), y2.clamp(0.0, fh)];
                if c[2] - c[0] <= 0.0 || c[3] - c[1] <= 0.0 {
                    dropped += 1;
                    continue;
                }
                out.push(BBox::from_pixel_xyxy(class, c, w, h));
            }
            if dropped > 0 {
                issues.push(issue(
                    IssueKind::LabelWarning,
                    &c.image,
                    format!("{dropped} zero-size box(es) dropped"),
                ));
            }
            (Some(ImageEntry { image: c.image, label: None, size: Some((w, h)), boxes: out }), issues, fresh)
        }
    }
}

/// `class0`, `class1`, ... for datasets without names.
pub(crate) fn default_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("class{i}")).collect()
}
