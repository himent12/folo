//! Datasets: the [`Dataset`] trait, YOLO and COCO readers with format auto-detection, and the load report.

mod coco;
mod labels;
mod yolo;

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

    /// Images skipped because of a broken label.
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
    let path = path.as_ref();
    if !path.exists() {
        return Err(DataError::io(
            path,
            std::io::Error::new(std::io::ErrorKind::NotFound, "dataset path not found"),
        ));
    }
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase());
    if path.is_file() {
        return match (opts.format, ext.as_deref()) {
            (Format::Auto | Format::Yolo, Some("yaml" | "yml")) => yolo::open_yaml(path, opts),
            (Format::Auto | Format::Coco, Some("json")) => {
                let parent = path.parent().unwrap_or(Path::new("."));
                let root =
                    if parent.ends_with("annotations") { parent.parent().unwrap_or(parent) } else { parent };
                coco::open_jsons(root, vec![(coco::guess_split(path), path.to_path_buf())], opts)
            }
            _ => Err(DataError::Format(format!(
                "{} is not a data.yaml or a COCO .json file (format {:?})",
                path.display(),
                opts.format
            ))),
        };
    }
    let want_yolo = matches!(opts.format, Format::Auto | Format::Yolo);
    let want_coco = matches!(opts.format, Format::Auto | Format::Coco);
    if want_yolo && let Some(yaml) = yolo::find_yaml(path) {
        return yolo::open_yaml(&yaml, opts);
    }
    if want_coco {
        let jsons = coco::find_jsons(path);
        if !jsons.is_empty() {
            return coco::open_jsons(path, jsons, opts);
        }
    }
    if want_yolo && yolo::looks_like_yolo_dir(path) {
        return yolo::open_dir(path, opts);
    }
    Err(DataError::Format(format!(
        "could not detect a dataset in {}: expected a data.yaml, COCO instances JSON, or images/ + labels/ folders",
        path.display()
    )))
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
    let results: Vec<(Option<ImageEntry>, Vec<Issue>)> =
        candidates.into_par_iter().map(|c| check_candidate(c, num_classes, opts)).collect();
    let mut entries = Vec::with_capacity(results.len());
    for (e, issues) in results {
        report.issues.extend(issues);
        entries.extend(e);
    }
    entries
}

fn check_candidate(
    c: Candidate,
    num_classes: Option<usize>,
    opts: &OpenOptions,
) -> (Option<ImageEntry>, Vec<Issue>) {
    let mut issues = Vec::new();
    let issue = |kind, path: &Path, message: String| Issue { kind, path: path.to_path_buf(), message };
    if !c.image.is_file() {
        issues.push(issue(IssueKind::MissingImage, &c.image, "image file not found".into()));
        return (None, issues);
    }
    let need_size = opts.verify_images || matches!(&c.labels, LabelSource::Coco { declared: None, .. });
    let size = if need_size {
        match probe_size(&c.image) {
            Ok(s) => Some(s),
            Err(e) => {
                issues.push(issue(IssueKind::CorruptImage, &c.image, e.to_string()));
                return (None, issues);
            }
        }
    } else {
        None
    };
    match c.labels {
        LabelSource::Yolo(label) => {
            let (boxes, label) = if label.is_file() {
                let text = match std::fs::read(&label) {
                    Ok(bytes) => match String::from_utf8(bytes) {
                        Ok(t) => t,
                        Err(_) => {
                            issues.push(issue(
                                IssueKind::BrokenLabel,
                                &label,
                                "label file is not UTF-8 text".into(),
                            ));
                            return (None, issues);
                        }
                    },
                    Err(e) => {
                        issues.push(issue(IssueKind::BrokenLabel, &label, e.to_string()));
                        return (None, issues);
                    }
                };
                match labels::parse_yolo(&text, num_classes) {
                    Ok(p) => {
                        for w in p.warnings {
                            issues.push(issue(IssueKind::LabelWarning, &label, w));
                        }
                        (p.boxes, Some(label))
                    }
                    Err((line, msg)) => {
                        issues.push(issue(IssueKind::BrokenLabel, &label, format!("line {line}: {msg}")));
                        return (None, issues);
                    }
                }
            } else {
                (Vec::new(), None)
            };
            (Some(ImageEntry { image: c.image, label, size, boxes }), issues)
        }
        LabelSource::Coco { boxes, declared, broken, label } => {
            if let Some(msg) = broken {
                issues.push(issue(IssueKind::BrokenLabel, &label, format!("{}: {msg}", c.image.display())));
                return (None, issues);
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
                return (None, issues);
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
            (Some(ImageEntry { image: c.image, label: None, size: Some((w, h)), boxes: out }), issues)
        }
    }
}

/// `class0`, `class1`, ... for datasets without names.
pub(crate) fn default_names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("class{i}")).collect()
}
