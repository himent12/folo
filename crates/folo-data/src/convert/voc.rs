//! Configurable Pascal VOC XML export ([`export_voc`]).
//!
//! Exports any loaded FOLO dataset (YOLO, COCO, or Pascal VOC) to standard Pascal VOC layout
//! (`Annotations/*.xml`, `ImageSets/Main/{split}.txt`, optional `JPEGImages/`, optional
//! `ImageSets/Main/trainval.txt`, and optional `classes.txt`) with selectable splits, `copy` / `link` / `none`
//! image handling, 1-based or 0-based `<bndbox>` coordinates, deterministic ordering, collision-safe exclusive
//! file reservation across splits, and atomic XML/manifest writes.

use crate::Split;
use crate::dataset::voc::{format_voc_xml_with, xml::valid_xml_char};
use crate::dataset::{ImageEntry, LoadedDataset, VocCoordMode};
use crate::error::{DataError, Result};
use crate::image::probe_size;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::io::{ErrorKind, Write as _};
use std::path::{Path, PathBuf};

/// Maximum UTF-8 byte length of a sanitized VOC stem before collision suffixes (`_<n>`) and extension
/// (`.xml`/`.<ext>`), leaving headroom under the 255-byte filesystem component limit.
const MAX_VOC_STEM_BYTES: usize = 200;

use super::{
    copy, deterministic_entries, image_extension, mkdir, resolve_link_source, select_splits,
    should_fallback_from_link, validate_common_entries, validate_single_component, write_file_atomic,
};

/// How [`export_voc`] handles source images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VocImageMode {
    /// Copy images byte for byte into `<out>/JPEGImages/` (default; self-contained).
    #[default]
    Copy,
    /// Hard-link images into `<out>/JPEGImages/`, falling back to a byte-for-byte copy if hard-linking
    /// is unsupported or crosses filesystems.
    Link,
    /// Do not copy or link images; write only `Annotations/*.xml`, `ImageSets/Main/*.txt`, and optional
    /// `classes.txt` (does not create `JPEGImages/`).
    None,
}

/// Options for [`export_voc`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct VocExportOptions {
    /// Splits to export. Empty exports every non-empty split in canonical `[Train, Val, Test]` order;
    /// non-empty lists are deduplicated and exported in canonical `[Train, Val, Test]` order.
    pub splits: Vec<Split>,
    /// How source images are materialized into `JPEGImages/`.
    pub images: VocImageMode,
    /// Pixel coordinate convention for `<bndbox>` elements (default [`VocCoordMode::OneBased`], the
    /// VOC2007/VOC2012 devkit standard).
    pub coord_mode: VocCoordMode,
    /// Write `<out>/classes.txt` when the dataset has at least one class (default `true`, preserving exact
    /// class ID ordering on re-import).
    pub write_classes_txt: bool,
    /// Also write `<out>/ImageSets/Main/trainval.txt` containing the union of exported `train` and `val`
    /// image IDs when both `train` and `val` are exported (default `false`).
    pub write_trainval: bool,
    /// Optional `<folder>` tag text written inside each XML annotation (default `"JPEGImages"`).
    pub folder: Option<String>,
}

impl Default for VocExportOptions {
    fn default() -> Self {
        Self {
            splits: Vec::new(),
            images: VocImageMode::Copy,
            coord_mode: VocCoordMode::OneBased,
            write_classes_txt: true,
            write_trainval: false,
            folder: None,
        }
    }
}

/// Per-split summary inside [`VocExportReport`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocSplitReport {
    /// Exported split.
    pub split: Split,
    /// Written split list file (`<out>/ImageSets/Main/<split>.txt`).
    pub image_set_file: PathBuf,
    /// Images exported in this split.
    pub images: usize,
    /// Annotations (boxes) exported in this split.
    pub annotations: usize,
    /// Images materialized via hard links (non-zero only under [`VocImageMode::Link`]).
    pub linked_images: usize,
    /// Images materialized via byte-for-byte copies (`Copy`, or `Link` fallback).
    pub copied_images: usize,
    /// Images whose files were not copied or linked ([`VocImageMode::None`]).
    pub skipped_images: usize,
}

/// Summary of what [`export_voc`] wrote.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VocExportReport {
    /// Output root directory.
    pub root: PathBuf,
    /// Image handling mode used.
    pub image_mode: VocImageMode,
    /// Coordinate convention written into `<bndbox>`.
    pub coord_mode: VocCoordMode,
    /// Number of classes in the dataset (`ds.classes.len()`).
    pub classes: usize,
    /// Path to `<out>/classes.txt` if written.
    pub classes_file: Option<PathBuf>,
    /// Path to `<out>/ImageSets/Main/trainval.txt` if written.
    pub trainval_file: Option<PathBuf>,
    /// Images written per split `[train, val, test]`.
    pub splits: [usize; 3],
    /// Total bounding-box annotations written across all exported splits.
    pub boxes: usize,
    /// Per-split details in canonical `[Train, Val, Test]` order.
    pub exported_splits: Vec<VocSplitReport>,
}

/// Exports `ds` to `out` in Pascal VOC XML format (`Annotations/*.xml`, `ImageSets/Main/<split>.txt`,
/// optional `JPEGImages/`, optional `ImageSets/Main/trainval.txt`, and optional `classes.txt`).
///
/// `out` must not exist or must be an empty directory: it is claimed with [`crate::unfinished::claim`] before
/// the first file is written, and XML/manifest files are written atomically via sibling temporary files and
/// rename. Validates all requested splits, class names, folder name, class IDs, and box coordinates before
/// writing anything.
pub fn export_voc(ds: &LoadedDataset, out: &Path, opts: &VocExportOptions) -> Result<VocExportReport> {
    let selected = select_splits(ds, &opts.splits)?;
    validate_voc_inputs(ds, &selected, opts)?;
    let folder = opts.folder.as_deref().unwrap_or("JPEGImages");

    let claim = crate::unfinished::claim(out)?;
    let ann_dir = out.join("Annotations");
    let sets_dir = out.join("ImageSets").join("Main");
    let img_dir = out.join("JPEGImages");
    mkdir(&ann_dir)?;
    mkdir(&sets_dir)?;
    if opts.images != VocImageMode::None {
        mkdir(&img_dir)?;
    }

    let mut taken = HashSet::new();
    let mut splits = [0usize; 3];
    let mut total_boxes = 0usize;
    let mut exported_splits = Vec::with_capacity(selected.len());
    let mut train_stems: Option<Vec<String>> = None;
    let mut val_stems: Option<Vec<String>> = None;

    for split in selected {
        let d = ds.split(split);
        let ordered = deterministic_entries(d.entries());

        // Probe missing or zero dimensions in parallel while enforcing the read guard.
        let sizes: Vec<(u32, u32)> = ordered
            .par_iter()
            .map(|e| e.size.filter(|&(w, h)| w > 0 && h > 0).map_or_else(|| probe_size(&e.image), Ok))
            .collect::<Result<_>>()?;

        let (reserved, linked_images, copied_images, skipped_images) =
            reserve_and_materialize(&ann_dir, &img_dir, &ordered, opts.images, &mut taken)?;

        ordered.par_iter().zip(&reserved).zip(&sizes).try_for_each(|((e, target), &(w, h))| {
            let xml =
                format_voc_xml_with(&target.file_name, folder, w, h, &ds.classes, &e.boxes, opts.coord_mode);
            let xml_path = ann_dir.join(format!("{}.xml", target.stem));
            write_text_atomic(&xml_path, &xml)
        })?;

        let mut list = String::with_capacity(reserved.len() * 16);
        let mut stems = Vec::with_capacity(reserved.len());
        for target in &reserved {
            let _ = writeln!(list, "{}", target.stem);
            stems.push(target.stem.clone());
        }
        let image_set_file = sets_dir.join(format!("{split}.txt"));
        write_text_atomic(&image_set_file, &list)?;

        if split == Split::Train {
            train_stems = Some(stems);
        } else if split == Split::Val {
            val_stems = Some(stems);
        }

        let n_imgs = ordered.len();
        let n_anns: usize = ordered.iter().map(|e| e.boxes.len()).sum();
        splits[split as usize] = n_imgs;
        total_boxes += n_anns;
        exported_splits.push(VocSplitReport {
            split,
            image_set_file,
            images: n_imgs,
            annotations: n_anns,
            linked_images,
            copied_images,
            skipped_images,
        });
    }

    let trainval_file = if opts.write_trainval
        && let (Some(train), Some(val)) = (train_stems, val_stems)
    {
        let mut combined = String::with_capacity((train.len() + val.len()) * 16);
        for stem in train.iter().chain(val.iter()) {
            let _ = writeln!(combined, "{stem}");
        }
        let p = sets_dir.join("trainval.txt");
        write_text_atomic(&p, &combined)?;
        Some(p)
    } else {
        None
    };

    let classes_file = if opts.write_classes_txt && !ds.classes.is_empty() {
        let mut names_txt = String::new();
        for c in &ds.classes {
            let _ = writeln!(names_txt, "{c}");
        }
        let p = out.join("classes.txt");
        write_text_atomic(&p, &names_txt)?;
        Some(p)
    } else {
        None
    };

    claim.finish()?;
    Ok(VocExportReport {
        root: out.to_path_buf(),
        image_mode: opts.images,
        coord_mode: opts.coord_mode,
        classes: ds.classes.len(),
        classes_file,
        trainval_file,
        splits,
        boxes: total_boxes,
        exported_splits,
    })
}

struct ReservedVoc {
    stem: String,
    file_name: String,
}

fn validate_voc_inputs(ds: &LoadedDataset, splits: &[Split], opts: &VocExportOptions) -> Result<()> {
    let mut seen_classes: HashMap<&str, usize> = HashMap::with_capacity(ds.classes.len());
    for (i, c) in ds.classes.iter().enumerate() {
        if c.trim().is_empty()
            || c.trim() != c
            || c.starts_with('\u{feff}')
            || c.contains(['\n', '\r'])
            || !c.chars().all(|ch| valid_xml_char(ch as u32).is_some())
        {
            return Err(DataError::Config(format!(
                "cannot export class #{i} ({c:?}) to Pascal VOC: class names must be non-empty, trimmed, single-line, BOM-free, valid XML 1.0 text"
            )));
        }
        if let Some(&first) = seen_classes.get(c.as_str()) {
            return Err(DataError::Config(format!(
                "cannot export duplicate class name {c:?} (class #{first} and class #{i}) to Pascal VOC: distinct class IDs would collapse"
            )));
        }
        seen_classes.insert(c.as_str(), i);
    }
    if let Some(folder) = opts.folder.as_deref()
        && (folder.trim().is_empty()
            || folder.trim() != folder
            || folder.contains(['\n', '\r'])
            || !folder.chars().all(|ch| valid_xml_char(ch as u32).is_some()))
    {
        return Err(DataError::Config(format!(
            "invalid Pascal VOC <folder> value ({folder:?}): must be non-empty, trimmed, single-line, valid XML 1.0 text"
        )));
    }
    validate_common_entries(ds, splits)
}

/// Sanitizes a source image path into a whitespace-free, `#`-free, dot-free, BOM-free, bounded-length,
/// valid XML 1.0 VOC stem and a safe image extension so `ImageSets/Main/*.txt` IDs and `<filename>` tags
/// round-trip cleanly across FOLO and external Pascal VOC tools.
fn sanitize_voc_stem_and_ext(p: &Path) -> (String, String) {
    let raw_stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut stem: String = raw_stem
        .chars()
        .map(|c| {
            if c.is_whitespace()
                || matches!(c, '.' | '/' | '\\' | ':' | '\0' | '\u{feff}')
                || c.is_control()
                || valid_xml_char(c as u32).is_none()
            {
                '_'
            } else {
                c
            }
        })
        .collect();
    while stem.starts_with('#') {
        stem.replace_range(..1, "_");
    }
    if stem.len() > MAX_VOC_STEM_BYTES {
        let mut end = MAX_VOC_STEM_BYTES;
        while end > 0 && !stem.is_char_boundary(end) {
            end -= 1;
        }
        stem.truncate(end);
    }
    if stem.is_empty() {
        stem = "image".into();
    }
    let raw_ext = image_extension(p);
    let ext: String = raw_ext.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let ext = if ext.is_empty() { "jpg".to_string() } else { ext };
    (stem, ext)
}

/// Picks the next VOC `(stem, file_name)` not yet in `taken` (case-insensitively across all splits) and validates
/// both `<stem>.xml` and `<stem>.<ext>` as safe single path components.
fn next_voc_candidate(
    base_stem: &str,
    ext: &str,
    taken: &mut HashSet<String>,
    n: &mut usize,
) -> Result<(String, String)> {
    loop {
        let stem = if *n == 0 { base_stem.to_string() } else { format!("{base_stem}_{}", *n) };
        *n += 1;
        if taken.insert(stem.to_lowercase()) {
            let file_name = format!("{stem}.{ext}");
            let xml_name = format!("{stem}.xml");
            validate_single_component(&file_name)?;
            validate_single_component(&xml_name)?;
            return Ok((stem, file_name));
        }
    }
}

/// Exclusively reserves `<ann_dir>/<stem>.xml` (and `<img_dir>/<file_name>` when materializing images) and copies
/// or hard-links images in parallel, retrying on `AlreadyExists` so case- or normalization-insensitive
/// filesystems never overwrite another entry across splits or mismatched extensions.
fn reserve_and_materialize(
    ann_dir: &Path,
    img_dir: &Path,
    ordered: &[&ImageEntry],
    mode: VocImageMode,
    taken: &mut HashSet<String>,
) -> Result<(Vec<ReservedVoc>, usize, usize, usize)> {
    let mut reserved = Vec::with_capacity(ordered.len());
    let mut copy_jobs: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut linked = 0usize;

    for e in ordered {
        let (base_stem, ext) = sanitize_voc_stem_and_ext(&e.image);
        let link_src = if mode == VocImageMode::Link { Some(resolve_link_source(&e.image)?) } else { None };
        let mut n = 0usize;
        loop {
            let (stem, file_name) = next_voc_candidate(&base_stem, &ext, taken, &mut n)?;
            let xml_dest = ann_dir.join(format!("{stem}.xml"));
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&xml_dest) {
                Ok(_xml_placeholder) => {}
                Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(DataError::io(&xml_dest, err)),
            }

            match mode {
                VocImageMode::None => {
                    reserved.push(ReservedVoc { stem, file_name });
                    break;
                }
                VocImageMode::Copy => {
                    let img_dest = img_dir.join(&file_name);
                    match std::fs::OpenOptions::new().write(true).create_new(true).open(&img_dest) {
                        Ok(_img_placeholder) => {
                            copy_jobs.push((e.image.clone(), img_dest));
                            reserved.push(ReservedVoc { stem, file_name });
                            break;
                        }
                        Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                            let _ = std::fs::remove_file(&xml_dest);
                            continue;
                        }
                        Err(err) => {
                            let _ = std::fs::remove_file(&xml_dest);
                            return Err(DataError::io(&img_dest, err));
                        }
                    }
                }
                VocImageMode::Link => {
                    let src = link_src.as_ref().expect("resolved for Link mode");
                    let img_dest = img_dir.join(&file_name);
                    match std::fs::hard_link(src, &img_dest) {
                        Ok(()) => {
                            linked += 1;
                            reserved.push(ReservedVoc { stem, file_name });
                            break;
                        }
                        Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                            let _ = std::fs::remove_file(&xml_dest);
                            continue;
                        }
                        Err(err) if should_fallback_from_link(&err) => {
                            match std::fs::OpenOptions::new().write(true).create_new(true).open(&img_dest) {
                                Ok(_img_placeholder) => {
                                    copy_jobs.push((src.clone(), img_dest));
                                    reserved.push(ReservedVoc { stem, file_name });
                                    break;
                                }
                                Err(ce) if ce.kind() == ErrorKind::AlreadyExists => {
                                    let _ = std::fs::remove_file(&xml_dest);
                                    continue;
                                }
                                Err(ce) => {
                                    let _ = std::fs::remove_file(&xml_dest);
                                    return Err(DataError::io(&img_dest, ce));
                                }
                            }
                        }
                        Err(err) => {
                            let _ = std::fs::remove_file(&xml_dest);
                            return Err(DataError::io(src, err));
                        }
                    }
                }
            }
        }
    }

    let copied = copy_jobs.len();
    copy_jobs.par_iter().try_for_each(|(src, dest)| copy(src, dest))?;
    let skipped = if mode == VocImageMode::None { ordered.len() } else { 0 };
    Ok((reserved, linked, copied, skipped))
}

fn write_text_atomic(path: &Path, text: &str) -> Result<()> {
    write_file_atomic(path, |w| w.write_all(text.as_bytes()).map_err(|e| DataError::io(path, e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voc_name_sanitizer_handles_spaces_dots_hashes_bom_long_stems_and_collisions() {
        let long_ascii = format!("g/{}.png", "a".repeat(251));
        let long_utf8 = format!("h/{}.png", "é".repeat(120));
        let paths = [
            Path::new("a/street scene.png"),
            Path::new("b/street scene.jpg"),
            Path::new("c/frame.jpg.png"),
            Path::new("d/#1.png"),
            Path::new("e/.."),
            Path::new("f/a&b'c.png"),
            Path::new("i/\u{feff}frame.png"),
            Path::new(long_ascii.as_str()),
            Path::new(long_ascii.as_str()),
            Path::new(long_utf8.as_str()),
        ];
        let mut taken = HashSet::new();
        let got: Vec<(String, String)> = paths
            .into_iter()
            .map(|p| {
                let (stem, ext) = sanitize_voc_stem_and_ext(p);
                let mut n = 0usize;
                next_voc_candidate(&stem, &ext, &mut taken, &mut n).unwrap()
            })
            .collect();
        assert_eq!(
            &got[..7],
            &[
                ("street_scene".into(), "street_scene.png".into()),
                ("street_scene_1".into(), "street_scene_1.jpg".into()),
                ("frame_jpg".into(), "frame_jpg.png".into()),
                ("_1".into(), "_1.png".into()),
                ("image".into(), "image.jpg".into()),
                ("a&b'c".into(), "a&b'c.png".into()),
                ("_frame".into(), "_frame.png".into()),
            ]
        );
        assert_eq!(got[7].0.len(), MAX_VOC_STEM_BYTES);
        assert_eq!(got[8].0.len(), MAX_VOC_STEM_BYTES + 2); // `_1` collision suffix
        assert!(got[8].1.len() <= 255);
        assert!(got[9].0.len() <= MAX_VOC_STEM_BYTES);
        assert!(got[9].1.len() <= 255);
    }

    #[test]
    fn voc_export_options_deserialize_with_defaults() {
        let opts: VocExportOptions = serde_json::from_str(r#"{"images": "link"}"#).unwrap();
        assert_eq!(opts.images, VocImageMode::Link);
        assert_eq!(opts.coord_mode, VocCoordMode::OneBased);
        assert!(opts.write_classes_txt);
        assert!(!opts.write_trainval);
        assert!(opts.splits.is_empty());
        assert!(opts.folder.is_none());
    }
}
