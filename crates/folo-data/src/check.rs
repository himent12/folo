//! Dataset validation (`folo dataset check`): every problem in a dataset, one [`CheckIssue`] per bad label line,
//! unreadable image or duplicate, each with the fix a cleanup would apply.
//!
//! Unlike [`crate::open`], which stops at the first bad line of a label file and skips the image, the checker reads
//! every line of every label file, fully decodes every image (header-only with [`CheckOptions::decode`] off) and
//! hashes file contents to find exact duplicates. It never modifies anything.

use crate::dataset::labels::{Line, LineError, parse_line};
use crate::dataset::{IssueKind, OpenOptions, Source, detect, label_path_for, open_with, yolo};
use crate::error::Result;
use crate::{Format, Split};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// What kind of problem a [`CheckIssue`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    /// A label line that is not `class cx cy w h` or a polygon of finite numbers, or an unreadable label file.
    BrokenLabel,
    /// A class id at or above the number of classes (or not a non-negative integer).
    ClassOverflow,
    /// A box clearly outside `[0, 1]` (pixel coordinates, not normalised) or with a negative size.
    OutOfRangeBox,
    /// A box slightly outside the image (within the 0.01 tolerance); the loader clips it.
    ClippedBox,
    /// A box with no area inside the image; the loader drops it.
    ZeroSizeBox,
    /// The same box twice in one label file; the loader keeps one.
    DuplicateLabel,
    /// An image file that exists but cannot be read or decoded.
    CorruptImage,
    /// An image named by an image list or COCO JSON that does not exist.
    MissingImage,
    /// An image whose bytes are identical to an earlier image's.
    DuplicateImage,
    /// An image without boxes (a background image). Informational.
    EmptyImage,
    /// Anything else (a split path that does not exist, a size mismatch in COCO JSON, ...).
    Other,
}

impl CheckKind {
    /// The kind's snake_case name, as serialised.
    pub fn as_str(self) -> &'static str {
        match self {
            CheckKind::BrokenLabel => "broken_label",
            CheckKind::ClassOverflow => "class_overflow",
            CheckKind::OutOfRangeBox => "out_of_range_box",
            CheckKind::ClippedBox => "clipped_box",
            CheckKind::ZeroSizeBox => "zero_size_box",
            CheckKind::DuplicateLabel => "duplicate_label",
            CheckKind::CorruptImage => "corrupt_image",
            CheckKind::MissingImage => "missing_image",
            CheckKind::DuplicateImage => "duplicate_image",
            CheckKind::EmptyImage => "empty_image",
            CheckKind::Other => "other",
        }
    }

    /// How serious this kind is.
    pub fn severity(self) -> Severity {
        match self {
            CheckKind::BrokenLabel
            | CheckKind::ClassOverflow
            | CheckKind::OutOfRangeBox
            | CheckKind::CorruptImage
            | CheckKind::MissingImage => Severity::Error,
            CheckKind::ClippedBox
            | CheckKind::ZeroSizeBox
            | CheckKind::DuplicateLabel
            | CheckKind::DuplicateImage
            | CheckKind::Other => Severity::Warning,
            CheckKind::EmptyImage => Severity::Info,
        }
    }
}

/// How serious an issue is. Errors make the loader skip an image; warnings are fixed up silently while loading
/// (or are worth a look); infos are normal (background images).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// The image is skipped by the loader.
    Error,
    /// Fixed up on load, or suspicious.
    Warning,
    /// Not a problem.
    Info,
}

/// One problem.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckIssue {
    /// What is wrong.
    pub kind: CheckKind,
    /// The file it is in (a label file for label problems, else the image or the JSON).
    pub file: PathBuf,
    /// 1-based line in `file`, for label-line problems.
    pub line: Option<usize>,
    /// Human-readable details.
    pub message: String,
    /// What a cleanup would do about it (`None` for infos).
    pub fix: Option<String>,
}

impl CheckIssue {
    fn new(kind: CheckKind, file: &Path, line: Option<usize>, message: String, fix: Option<String>) -> Self {
        CheckIssue { kind, file: file.to_path_buf(), line, message, fix }
    }

    /// The issue's severity.
    pub fn severity(&self) -> Severity {
        self.kind.severity()
    }
}

/// Options for [`check`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckOptions {
    /// Format to read; `Auto` detects it.
    pub format: Format,
    /// Decode every image fully (default). Off, only the header is read, which misses truncated image data.
    pub decode: bool,
}

impl Default for CheckOptions {
    fn default() -> Self {
        CheckOptions { format: Format::Auto, decode: true }
    }
}

/// The result of [`check`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CheckReport {
    /// The format that was read.
    pub format: Format,
    /// Dataset root.
    pub root: PathBuf,
    /// Class names (empty when the dataset declares none).
    pub classes: Vec<String>,
    /// Images examined per split `[train, val, test]`. YOLO counts every image file (broken ones too); COCO counts
    /// the images the loader kept (dropped ones are `missing_image` / `broken_label` issues).
    pub splits: [usize; 3],
    /// Every issue, grouped by image in dataset order.
    pub issues: Vec<CheckIssue>,
}

impl CheckReport {
    /// Total images examined.
    pub fn images(&self) -> usize {
        self.splits.iter().sum()
    }

    /// Issues of a severity.
    pub fn count(&self, severity: Severity) -> usize {
        self.issues.iter().filter(|i| i.severity() == severity).count()
    }

    /// True when there are no errors and no warnings (infos such as empty images are fine).
    pub fn is_clean(&self) -> bool {
        self.issues.iter().all(|i| i.severity() == Severity::Info)
    }

    /// Number of issues per kind, in kind order.
    pub fn counts(&self) -> BTreeMap<CheckKind, usize> {
        let mut m = BTreeMap::new();
        for i in &self.issues {
            *m.entry(i.kind).or_insert(0) += 1;
        }
        m
    }
}

/// Checks the dataset at `path` (anything [`crate::open`] accepts). Fails only when there is no readable dataset
/// description at all; every problem inside the dataset is an issue in the report.
pub fn check(path: impl AsRef<Path>, opts: &CheckOptions) -> Result<CheckReport> {
    let path = path.as_ref();
    match detect(path, opts.format)? {
        Source::Yaml(yaml) => check_yolo(yolo::scan_yaml(&yaml)?, opts),
        Source::YoloDir(root) => check_yolo(yolo::scan_dir(&root), opts),
        Source::Coco { .. } => check_coco(path, opts),
    }
}

/// What reading one image found.
struct ImageCheck {
    issues: Vec<CheckIssue>,
    /// Content hash and length, when the file was readable.
    hash: Option<(u128, u64)>,
    /// `(width, height)` after EXIF orientation, when the file was readable.
    size: Option<(u32, u32)>,
}

fn check_yolo(scan: yolo::YoloScan, opts: &CheckOptions) -> Result<CheckReport> {
    let nc = scan.names.as_ref().map(Vec::len);
    let mut issues: Vec<CheckIssue> = scan.report.issues.iter().map(from_load_issue).collect();
    let mut splits = [0usize; 3];
    let images: Vec<(Split, &PathBuf)> = Split::ALL
        .into_iter()
        .flat_map(|s| scan.splits.get(&s).into_iter().flatten().map(move |p| (s, p)))
        .collect();
    for (s, _) in &images {
        splits[*s as usize] += 1;
    }
    let results: Vec<ImageCheck> = images
        .par_iter()
        .map(|(_, image)| {
            let mut c = read_image_file(image, opts);
            let readable = c.hash.is_some();
            // Label problems are reported for unreadable images too, "empty" only for readable ones.
            c.issues.extend(
                lint_label_file(image, &label_path_for(image), nc)
                    .into_iter()
                    .filter(|i| readable || i.kind != CheckKind::EmptyImage),
            );
            c
        })
        .collect();
    let classes = scan.names.unwrap_or_default();
    merge(&mut issues, images.iter().map(|(_, p)| p.as_path()), results, &scan.root);
    Ok(CheckReport { format: Format::Yolo, root: scan.root, classes, splits, issues })
}

fn check_coco(path: &Path, opts: &CheckOptions) -> Result<CheckReport> {
    let ds = open_with(path, &OpenOptions { format: Format::Coco, verify_images: false })?;
    let mut issues: Vec<CheckIssue> = ds.report.issues.iter().map(from_load_issue).collect();
    let mut splits = [0usize; 3];
    let mut images = Vec::new();
    for s in Split::ALL {
        let entries = ds.split(s).entries();
        splits[s as usize] = entries.len();
        images.extend(entries.iter());
    }
    let results: Vec<ImageCheck> = images
        .par_iter()
        .map(|e| {
            let mut c = read_image_file(&e.image, opts);
            // Without image verification the loader keeps the JSON's size; boxes were normalised with it.
            if let (Some(declared), Some(actual)) = (e.size, c.size)
                && declared != actual
            {
                c.issues.push(CheckIssue::new(
                    CheckKind::Other,
                    &e.image,
                    None,
                    format!(
                        "JSON says {}x{} but the image is {}x{}",
                        declared.0, declared.1, actual.0, actual.1
                    ),
                    Some(format!("set width {} and height {} in the JSON", actual.0, actual.1)),
                ));
            }
            if e.boxes.is_empty() && c.hash.is_some() {
                c.issues.push(CheckIssue::new(
                    CheckKind::EmptyImage,
                    &e.image,
                    None,
                    "no annotations (background image)".into(),
                    None,
                ));
            }
            c
        })
        .collect();
    merge(&mut issues, images.iter().map(|e| e.image.as_path()), results, &ds.root);
    Ok(CheckReport { format: Format::Coco, root: ds.root, classes: ds.classes, splits, issues })
}

/// Appends per-image issues in order and flags images whose bytes repeat an earlier image's (named relative to
/// `root` with `/` separators, as issue files are reported).
fn merge<'a>(
    issues: &mut Vec<CheckIssue>,
    paths: impl Iterator<Item = &'a Path>,
    results: Vec<ImageCheck>,
    root: &Path,
) {
    let mut seen: HashMap<(u128, u64), &Path> = HashMap::new();
    for (path, r) in paths.zip(results) {
        issues.extend(r.issues);
        if let Some(key) = r.hash {
            match seen.get(&key) {
                Some(first) => issues.push(CheckIssue::new(
                    CheckKind::DuplicateImage,
                    path,
                    None,
                    format!("same bytes as {}", crate::paths::relative_to(first, root)),
                    Some(format!(
                        "remove this copy (and its label file); keep {}",
                        crate::paths::relative_to(first, root)
                    )),
                )),
                None => {
                    seen.insert(key, path);
                }
            }
        }
    }
}

fn from_load_issue(i: &crate::Issue) -> CheckIssue {
    let (kind, fix) = match i.kind {
        IssueKind::CorruptImage => (CheckKind::CorruptImage, "remove the image and its labels"),
        IssueKind::MissingImage => (CheckKind::MissingImage, "remove the entry that names the image"),
        IssueKind::BrokenLabel => (CheckKind::BrokenLabel, "fix or remove the broken annotation"),
        IssueKind::LabelWarning => (CheckKind::ZeroSizeBox, "remove the degenerate annotation"),
        IssueKind::Other => (CheckKind::Other, "fix the dataset description"),
    };
    CheckIssue::new(kind, &i.path, None, i.message.clone(), Some(fix.into()))
}

/// Reads an image: missing / unreadable / undecodable files are issues; readable ones get a content hash.
fn read_image_file(image: &Path, opts: &CheckOptions) -> ImageCheck {
    let corrupt = |msg: String| ImageCheck {
        issues: vec![CheckIssue::new(
            CheckKind::CorruptImage,
            image,
            None,
            msg,
            Some("remove the image and its label file".into()),
        )],
        hash: None,
        size: None,
    };
    let bytes = match crate::guard::check(image).and_then(|_| std::fs::read(image)) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return ImageCheck {
                issues: vec![CheckIssue::new(
                    CheckKind::MissingImage,
                    image,
                    None,
                    "image file not found".into(),
                    Some("remove the entry that names the image".into()),
                )],
                hash: None,
                size: None,
            };
        }
        Err(e) => return corrupt(format!("cannot read: {e}")),
    };
    let size = if opts.decode {
        match crate::image::decode(&bytes) {
            // Decoders fill a JPEG cut off mid-scan with grey instead of failing.
            Ok(_) if crate::image::jpeg_truncated(&bytes) => Err(format!(
                "truncated JPEG: the data ends before the end-of-image marker ({} bytes; it decodes with grey                  where the data is missing)",
                bytes.len()
            )),
            Ok(i) => Ok((i.width, i.height)),
            Err(crate::DataError::Image { message, .. }) => Err(format!("cannot decode: {message}")),
            Err(e) => Err(format!("cannot decode: {e}")),
        }
    } else {
        image::ImageReader::new(std::io::Cursor::new(&bytes))
            .with_guessed_format()
            .map_err(|e| e.to_string())
            .and_then(|r| r.into_dimensions().map_err(|e| e.to_string()))
            .and_then(|(w, h)| if w == 0 || h == 0 { Err("image has zero size".into()) } else { Ok((w, h)) })
            // The header has the stored size; EXIF orientations 5-8 swap it (decode applies them).
            .map(
                |(w, h)| {
                    if crate::jpeg_exif_orientation(&bytes).is_some_and(|o| o >= 5) { (h, w) } else { (w, h) }
                },
            )
    };
    match size {
        Ok(size) => ImageCheck {
            issues: Vec::new(),
            hash: Some((hash128(&bytes), bytes.len() as u64)),
            size: Some(size),
        },
        Err(e) if opts.decode => corrupt(e),
        Err(e) => corrupt(format!("cannot decode: {}", e.trim())),
    }
}

/// Every problem in one YOLO label file. A missing or box-less file is an [`CheckKind::EmptyImage`] of `image`.
fn lint_label_file(image: &Path, label: &Path, nc: Option<usize>) -> Vec<CheckIssue> {
    let mut out = Vec::new();
    let bytes = match crate::guard::check(label).and_then(|_| std::fs::read(label)) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            out.push(CheckIssue::new(
                CheckKind::EmptyImage,
                image,
                None,
                "no label file (background image)".into(),
                None,
            ));
            return out;
        }
        Err(e) => {
            out.push(CheckIssue::new(
                CheckKind::BrokenLabel,
                label,
                None,
                format!("cannot read: {e}"),
                Some("rewrite or delete the label file".into()),
            ));
            return out;
        }
    };
    let Ok(text) = String::from_utf8(bytes) else {
        out.push(CheckIssue::new(
            CheckKind::BrokenLabel,
            label,
            None,
            "label file is not UTF-8 text".into(),
            Some("rewrite or delete the label file".into()),
        ));
        return out;
    };
    let (lint, boxes) = lint_labels(&text, nc);
    out.extend(
        lint.into_iter()
            .map(|(kind, line, message, fix)| CheckIssue::new(kind, label, Some(line), message, Some(fix))),
    );
    if boxes == 0 && out.is_empty() {
        out.push(CheckIssue::new(
            CheckKind::EmptyImage,
            image,
            None,
            "label file has no boxes (background image)".into(),
            None,
        ));
    }
    out
}

/// A line-level issue: `(kind, 1-based line, message, fix)`.
type LineIssue = (CheckKind, usize, String, String);

/// Line-level issues of YOLO label text, and the number of distinct usable boxes.
fn lint_labels(text: &str, nc: Option<usize>) -> (Vec<LineIssue>, usize) {
    let mut out = Vec::new();
    let mut vals = Vec::with_capacity(16);
    let mut kept: Vec<(crate::BBox, usize)> = Vec::new();
    let broken = " (the loader skips this image)";
    for (i, raw) in text.lines().enumerate() {
        let ln = i + 1;
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        match parse_line(line, nc, &mut vals) {
            Err((e, msg)) => {
                let kind = match e {
                    LineError::Malformed => CheckKind::BrokenLabel,
                    LineError::ClassId => CheckKind::ClassOverflow,
                    LineError::OutOfRange => CheckKind::OutOfRangeBox,
                };
                out.push((kind, ln, format!("{msg}{broken}"), format!("delete line {ln}")));
            }
            Ok(Line::Degenerate { .. }) => out.push((
                CheckKind::ZeroSizeBox,
                ln,
                "box has no area inside the image".into(),
                format!("delete line {ln}"),
            )),
            Ok(Line::Box { b, clipped }) => {
                if clipped {
                    out.push((
                        CheckKind::ClippedBox,
                        ln,
                        "box extends slightly past the image edge".into(),
                        format!("clip the box on line {ln} to the image"),
                    ));
                }
                if let Some((_, first)) = kept.iter().find(|(k, _)| *k == b) {
                    out.push((
                        CheckKind::DuplicateLabel,
                        ln,
                        format!("same box as line {first}"),
                        format!("delete line {ln}"),
                    ));
                } else {
                    kept.push((b, ln));
                }
            }
        }
    }
    (out, kept.len())
}

/// A fast 128-bit content hash (two independent 64-bit multiply-rotate lanes over 8-byte words).
fn hash128(bytes: &[u8]) -> u128 {
    const K1: u64 = 0x9e37_79b9_7f4a_7c15;
    const K2: u64 = 0xc2b2_ae3d_27d4_eb4f;
    let (mut a, mut b) = (0x243f_6a88_85a3_08d3_u64 ^ bytes.len() as u64, 0x1319_8a2e_0370_7344_u64);
    let (words, tail) = bytes.as_chunks::<8>();
    for w in words {
        let v = u64::from_le_bytes(*w);
        a = (a ^ v).wrapping_mul(K1).rotate_left(29);
        b = (b ^ v.rotate_left(32)).wrapping_mul(K2).rotate_left(31);
    }
    let mut last = [0u8; 8];
    last[..tail.len()].copy_from_slice(tail);
    let v = u64::from_le_bytes(last) ^ ((tail.len() as u64) << 56);
    a = crate::splitmix64(&mut (a ^ v));
    b = crate::splitmix64(&mut (b ^ v.wrapping_mul(K1)));
    ((a as u128) << 64) | b as u128
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_lint() {
        let text = "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n3 0.5 0.5 0.1 0.1\n0 50 50 10 10\nx\n\n0 0.5 0.5 0 0.1\n\
                    0 0.995 0.5 0.02 0.1\n";
        let (issues, boxes) = lint_labels(text, Some(2));
        assert_eq!(boxes, 2);
        let kinds: Vec<(CheckKind, usize)> = issues.into_iter().map(|(k, l, ..)| (k, l)).collect();
        assert_eq!(
            kinds,
            vec![
                (CheckKind::DuplicateLabel, 2),
                (CheckKind::ClassOverflow, 3),
                (CheckKind::OutOfRangeBox, 4),
                (CheckKind::BrokenLabel, 5),
                (CheckKind::ZeroSizeBox, 7),
                (CheckKind::ClippedBox, 8),
            ]
        );
        assert_eq!(lint_labels("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.3 0.1 0.3 0.3\n", Some(2)), (vec![], 2));
    }

    #[test]
    fn hash_distinguishes_close_inputs() {
        assert_ne!(hash128(b"abc"), hash128(b"abd"));
        assert_ne!(hash128(b""), hash128(b"\0"));
        assert_ne!(hash128(&[0u8; 8]), hash128(&[0u8; 9]));
        assert_eq!(hash128(b"same bytes here!"), hash128(b"same bytes here!"));
    }
}
