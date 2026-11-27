//! Spec-compliant COCO instances JSON export ([`export_coco`]).
//!
//! Exports any loaded FOLO dataset (YOLO, COCO, or Pascal VOC) to standard COCO instances JSON
//! (`annotations/instances_<split>.json`) with selectable splits, `copy` / `link` / `reference` image handling,
//! deterministic 1-based IDs, typed streaming atomic JSON writes, and safe path handling.

use crate::Split;
use crate::dataset::{ImageEntry, LoadedDataset};
use crate::error::{DataError, Result};
use crate::image::probe_size;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};

use super::{
    copy, deterministic_entries, image_extension, mkdir, resolve_link_source, select_splits,
    should_fallback_from_link, validate_common_entries, validate_single_component, write_file_atomic,
};

/// How [`export_coco`] handles source images.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CocoImageMode {
    /// Copy images byte for byte into `<out>/images/<split>/` (default; self-contained).
    #[default]
    Copy,
    /// Hard-link images into `<out>/images/<split>/`, falling back to a byte-for-byte copy if hard-linking
    /// is unsupported or crosses filesystems.
    Link,
    /// Do not copy or link images; write `annotations/instances_<split>.json` referencing the source images
    /// via relative paths from `<out>/annotations/` (falling back to normalized absolute paths only when source
    /// and destination reside on different filesystem roots).
    Reference,
}

/// Options for [`export_coco`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CocoExportOptions {
    /// Splits to export. Empty exports every non-empty split in canonical `[Train, Val, Test]` order;
    /// non-empty lists are deduplicated and exported in canonical `[Train, Val, Test]` order.
    pub splits: Vec<Split>,
    /// How source images are materialized or referenced.
    pub images: CocoImageMode,
    /// Pretty-print JSON instead of compact single-line output.
    pub pretty: bool,
    /// Optional `info.description` string; defaults to `"exported by FOLO from <root>"`.
    pub description: Option<String>,
}

/// Per-split summary inside [`CocoExportReport`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CocoSplitReport {
    /// Exported split.
    pub split: Split,
    /// Written annotation file (`<out>/annotations/instances_<split>.json`).
    pub annotation_file: PathBuf,
    /// Images exported in this split.
    pub images: usize,
    /// Annotations (boxes) exported in this split.
    pub annotations: usize,
    /// Images materialized via hard links (non-zero only under [`CocoImageMode::Link`]).
    pub linked_images: usize,
    /// Images materialized via byte-for-byte copies (`Copy`, or `Link` fallback).
    pub copied_images: usize,
    /// Images referenced in place without copying or linking ([`CocoImageMode::Reference`]).
    pub referenced_images: usize,
}

/// Summary of what [`export_coco`] wrote.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CocoExportReport {
    /// Output root directory.
    pub root: PathBuf,
    /// Image handling mode used.
    pub image_mode: CocoImageMode,
    /// Number of categories written to each annotation file (`ds.classes.len()`).
    pub categories: usize,
    /// Images written per split `[train, val, test]`.
    pub splits: [usize; 3],
    /// Total bounding-box annotations written across all exported splits.
    pub boxes: usize,
    /// Per-split details in canonical `[Train, Val, Test]` order.
    pub exported_splits: Vec<CocoSplitReport>,
}

#[derive(Serialize)]
struct CocoDocument<'a> {
    info: CocoInfo<'a>,
    images: &'a [CocoImageEntry],
    annotations: &'a [CocoAnnotationEntry],
    categories: &'a [CocoCategoryEntry<'a>],
}

#[derive(Serialize)]
struct CocoInfo<'a> {
    description: &'a str,
}

#[derive(Serialize)]
struct CocoImageEntry {
    id: u64,
    file_name: String,
    width: u32,
    height: u32,
}

#[derive(Serialize)]
struct CocoAnnotationEntry {
    id: u64,
    image_id: u64,
    category_id: u32,
    bbox: [f64; 4],
    area: f64,
    iscrowd: u8,
}

#[derive(Serialize)]
struct CocoCategoryEntry<'a> {
    id: u32,
    name: &'a str,
    supercategory: &'static str,
}

/// Exports `ds` to `out` as spec-compliant COCO instances JSON (`annotations/instances_<split>.json`).
///
/// `out` must not exist or must be an empty directory: it is claimed with [`crate::unfinished::claim`] before the
/// first file is written, and each annotation JSON is written atomically via a sibling temporary file and rename.
/// Validates all requested splits, class IDs, and box coordinates before writing anything.
pub fn export_coco(ds: &LoadedDataset, out: &Path, opts: &CocoExportOptions) -> Result<CocoExportReport> {
    let selected = select_splits(ds, &opts.splits)?;
    validate_entries(ds, &selected, opts.images)?;

    let claim = crate::unfinished::claim(out)?;
    let ann_dir = out.join("annotations");
    mkdir(&ann_dir)?;

    let description = opts
        .description
        .clone()
        .unwrap_or_else(|| format!("exported by FOLO from {}", crate::paths::shown(&ds.root)));
    let categories: Vec<CocoCategoryEntry<'_>> = ds
        .classes
        .iter()
        .enumerate()
        .map(|(i, name)| CocoCategoryEntry { id: (i + 1) as u32, name: name.as_str(), supercategory: "" })
        .collect();

    let mut splits = [0usize; 3];
    let mut total_boxes = 0usize;
    let mut exported_splits = Vec::with_capacity(selected.len());

    for split in selected {
        let d = ds.split(split);
        let ordered = deterministic_entries(d.entries());

        // Probe missing or zero dimensions in parallel while enforcing the read guard.
        let sizes: Vec<(u32, u32)> = ordered
            .par_iter()
            .map(|e| e.size.filter(|&(w, h)| w > 0 && h > 0).map_or_else(|| probe_size(&e.image), Ok))
            .collect::<Result<_>>()?;

        let (file_names, linked_images, copied_images, referenced_images) = match opts.images {
            CocoImageMode::Copy => {
                let img_dir = out.join("images").join(split.as_str());
                mkdir(&img_dir)?;
                let names = reserve_exclusive_copy_targets(&img_dir, &ordered)?;
                ordered
                    .par_iter()
                    .zip(&names)
                    .try_for_each(|(e, name)| copy(&e.image, &img_dir.join(name)))?;
                let count = names.len();
                (names, 0, count, 0)
            }
            CocoImageMode::Link => {
                let img_dir = out.join("images").join(split.as_str());
                mkdir(&img_dir)?;
                let (names, linked, copied) = materialize_links(&img_dir, &ordered)?;
                (names, linked, copied, 0)
            }
            CocoImageMode::Reference => {
                // Anchor relative paths at `<out>/annotations/` (candidate #0 in `coco::image_base`) so both
                // `open(out)` and `open(out/annotations/instances_<split>.json)` resolve unambiguously even when
                // a same-named decoy exists under `<out>/`.
                let ann_base = canonical_existing(&ann_dir)?;
                let mut names = Vec::with_capacity(ordered.len());
                for e in &ordered {
                    crate::guard::check(&e.image).map_err(|err| DataError::io(&e.image, err))?;
                    names.push(reference_file_name(&ann_base, &e.image)?);
                }
                let count = names.len();
                (names, 0, 0, count)
            }
        };

        let mut images = Vec::with_capacity(ordered.len());
        let mut anns = Vec::with_capacity(ordered.iter().map(|e| e.boxes.len()).sum());
        for (i, ((e, file_name), &(w, h))) in ordered.iter().zip(file_names).zip(&sizes).enumerate() {
            let image_id = (i + 1) as u64;
            images.push(CocoImageEntry { id: image_id, file_name, width: w, height: h });
            for b in &e.boxes {
                let [raw_x1, raw_y1, raw_x2, raw_y2] = b.pixel_xyxy(w, h);
                let wf = w as f64;
                let hf = h as f64;
                let x1 = (raw_x1 as f64).clamp(0.0, wf);
                let y1 = (raw_y1 as f64).clamp(0.0, hf);
                let x2 = (raw_x2 as f64).clamp(x1, wf);
                let y2 = (raw_y2 as f64).clamp(y1, hf);
                let bw = x2 - x1;
                let bh = y2 - y1;
                anns.push(CocoAnnotationEntry {
                    id: (anns.len() + 1) as u64,
                    image_id,
                    category_id: b.class + 1,
                    bbox: [x1, y1, bw, bh],
                    area: bw * bh,
                    iscrowd: 0,
                });
            }
        }

        let ann_path = ann_dir.join(format!("instances_{split}.json"));
        let doc = CocoDocument {
            info: CocoInfo { description: &description },
            images: &images,
            annotations: &anns,
            categories: &categories,
        };
        write_json_atomic(&ann_path, &doc, opts.pretty)?;

        let n_imgs = images.len();
        let n_anns = anns.len();
        splits[split as usize] = n_imgs;
        total_boxes += n_anns;
        exported_splits.push(CocoSplitReport {
            split,
            annotation_file: ann_path,
            images: n_imgs,
            annotations: n_anns,
            linked_images,
            copied_images,
            referenced_images,
        });
    }

    claim.finish()?;
    Ok(CocoExportReport {
        root: out.to_path_buf(),
        image_mode: opts.images,
        categories: ds.classes.len(),
        splits,
        boxes: total_boxes,
        exported_splits,
    })
}

/// Validates class indices, box coordinates, and (in `Reference` mode) UTF-8 paths before claiming `out`.
fn validate_entries(ds: &LoadedDataset, splits: &[Split], image_mode: CocoImageMode) -> Result<()> {
    validate_common_entries(ds, splits)?;
    if image_mode == CocoImageMode::Reference {
        for &split in splits {
            for e in ds.split(split).entries() {
                if e.image.to_str().is_none() {
                    return Err(DataError::Config(format!(
                        "image path is not valid UTF-8 for COCO JSON: {}",
                        e.image.display()
                    )));
                }
            }
        }
    }
    Ok(())
}

/// Sanitizes a source image's stem and extension into safe single-component parts, stripping separators, drive
/// prefixes, NULs, and control characters.
fn sanitize_stem_and_ext(p: &Path) -> (String, String) {
    let raw_stem = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut stem: String = raw_stem
        .chars()
        .map(|c| if matches!(c, '/' | '\\' | ':' | '\0') || c.is_control() { '_' } else { c })
        .collect();
    if stem.is_empty() || stem == "." || stem == ".." || stem.chars().all(|c| c == '.') {
        stem = "image".into();
    }
    let raw_ext = image_extension(p);
    let ext: String = raw_ext.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    let ext = if ext.is_empty() { "jpg".to_string() } else { ext };
    (stem, ext)
}

/// Picks the next candidate filename not yet in `taken` (case-insensitively) and validates it as one path component.
fn next_candidate_name(stem: &str, ext: &str, taken: &mut HashSet<String>, n: &mut usize) -> Result<String> {
    loop {
        let candidate = if *n == 0 { stem.to_string() } else { format!("{stem}_{}", *n) };
        *n += 1;
        if taken.insert(candidate.to_lowercase()) {
            let file_name = format!("{candidate}.{ext}");
            validate_single_component(&file_name)?;
            return Ok(file_name);
        }
    }
}

/// Exclusively reserves destination filenames in `img_dir` via `create_new(true)` (dropping each handle
/// immediately to keep open FDs bounded at 1), retrying with a numeric suffix on `AlreadyExists` so
/// normalization- or case-insensitive filesystems (e.g. macOS APFS NFC/NFD) never overwrite another image.
fn reserve_exclusive_copy_targets(img_dir: &Path, ordered: &[&ImageEntry]) -> Result<Vec<String>> {
    let mut taken = HashSet::with_capacity(ordered.len());
    let mut names = Vec::with_capacity(ordered.len());
    for e in ordered {
        let (stem, ext) = sanitize_stem_and_ext(&e.image);
        let mut n = 0usize;
        loop {
            let file_name = next_candidate_name(&stem, &ext, &mut taken, &mut n)?;
            let dest = img_dir.join(&file_name);
            match std::fs::OpenOptions::new().write(true).create_new(true).open(&dest) {
                Ok(_placeholder) => {
                    names.push(file_name);
                    break;
                }
                Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(DataError::io(&dest, err)),
            }
        }
    }
    Ok(names)
}

/// Materializes images for `Link` mode with atomic exclusive naming: hard-links each resolved source file
/// (bumping the suffix on `AlreadyExists`), and reserves any cross-device/unsupported fallbacks exclusively
/// before copying them in parallel.
fn materialize_links(img_dir: &Path, ordered: &[&ImageEntry]) -> Result<(Vec<String>, usize, usize)> {
    let mut taken = HashSet::with_capacity(ordered.len());
    let mut names = Vec::with_capacity(ordered.len());
    let mut fallback_copies: Vec<(PathBuf, PathBuf)> = Vec::new();
    let mut linked = 0usize;

    for e in ordered {
        let src = resolve_link_source(&e.image)?;
        let (stem, ext) = sanitize_stem_and_ext(&e.image);
        let mut n = 0usize;
        loop {
            let file_name = next_candidate_name(&stem, &ext, &mut taken, &mut n)?;
            let dest = img_dir.join(&file_name);
            match std::fs::hard_link(&src, &dest) {
                Ok(()) => {
                    linked += 1;
                    names.push(file_name);
                    break;
                }
                Err(err) if err.kind() == ErrorKind::AlreadyExists => continue,
                Err(err) if should_fallback_from_link(&err) => {
                    let mut chosen = file_name;
                    let mut chosen_dest = dest;
                    loop {
                        match std::fs::OpenOptions::new().write(true).create_new(true).open(&chosen_dest) {
                            Ok(_placeholder) => break,
                            Err(ce) if ce.kind() == ErrorKind::AlreadyExists => {
                                chosen = next_candidate_name(&stem, &ext, &mut taken, &mut n)?;
                                chosen_dest = img_dir.join(&chosen);
                            }
                            Err(ce) => return Err(DataError::io(&chosen_dest, ce)),
                        }
                    }
                    names.push(chosen);
                    fallback_copies.push((src, chosen_dest));
                    break;
                }
                Err(err) => return Err(DataError::io(&src, err)),
            }
        }
    }

    let copied = fallback_copies.len();
    fallback_copies.par_iter().try_for_each(|(src, dest)| copy(src, dest))?;
    Ok((names, linked, copied))
}

/// Produces a path for `CocoImageMode::Reference` relative to `base` (`<out>/annotations/`) when possible, or a
/// normalized absolute path if `base` and `target` lie on different roots/drives. Propagates missing-file
/// canonicalization errors, verifies the resolved target is a regular file, and enforces the read guard on the
/// resolved target so symlinks cannot escape confinement.
fn reference_file_name(base: &Path, target: &Path) -> Result<String> {
    let target_norm = canonical_existing(target)?;
    crate::guard::check(&target_norm).map_err(|e| DataError::io(&target_norm, e))?;
    let meta = std::fs::metadata(&target_norm).map_err(|e| DataError::io(&target_norm, e))?;
    if !meta.is_file() {
        return Err(DataError::io(
            &target_norm,
            std::io::Error::new(ErrorKind::InvalidInput, "referenced image path is not a regular file"),
        ));
    }
    let chosen = diff_paths(&target_norm, base).unwrap_or(target_norm);
    let s = chosen.to_str().ok_or_else(|| {
        DataError::Config(format!("image path is not valid UTF-8 for COCO JSON: {}", target.display()))
    })?;
    if s.contains('\0') {
        return Err(DataError::Config(format!("image path contains NUL byte: {}", target.display())));
    }
    // Convert path separators only on Windows; on Unix `\` is a valid literal filename character.
    #[cfg(windows)]
    {
        Ok(s.replace('\\', "/"))
    }
    #[cfg(not(windows))]
    {
        Ok(s.to_string())
    }
}

fn canonical_existing(p: &Path) -> Result<PathBuf> {
    let canon = std::fs::canonicalize(p).map_err(|e| DataError::io(p, e))?;
    Ok(lexical_normalize(&strip_windows_verbatim(canon)))
}

#[cfg(windows)]
fn strip_windows_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        p
    }
}

#[cfg(not(windows))]
fn strip_windows_verbatim(p: PathBuf) -> PathBuf {
    p
}

fn lexical_normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                Some(Component::RootDir | Component::Prefix(_)) => {}
                _ => out.push(comp),
            },
            other => out.push(other),
        }
    }
    out
}

/// Relative path from `base` to `target`, or `None` if they have incompatible roots/prefixes.
fn diff_paths(target: &Path, base: &Path) -> Option<PathBuf> {
    let t: Vec<Component<'_>> = target.components().collect();
    let b: Vec<Component<'_>> = base.components().collect();
    let mut common = 0usize;
    while common < t.len() && common < b.len() && components_equal(t[common], b[common]) {
        common += 1;
    }
    if common == 0
        && (matches!(t.first(), Some(Component::Prefix(_) | Component::RootDir))
            || matches!(b.first(), Some(Component::Prefix(_) | Component::RootDir)))
    {
        return None;
    }
    let mut rel = PathBuf::new();
    for comp in &b[common..] {
        if matches!(comp, Component::Normal(_)) {
            rel.push("..");
        } else {
            return None;
        }
    }
    for comp in &t[common..] {
        rel.push(comp.as_os_str());
    }
    if rel.as_os_str().is_empty() {
        rel.push(".");
    }
    Some(rel)
}

fn components_equal(a: Component<'_>, b: Component<'_>) -> bool {
    #[cfg(windows)]
    {
        match (a, b) {
            // Canonical Windows paths preserve exact on-disk casing for normal components (important for
            // per-directory case-sensitive NTFS folders), while drive prefixes remain case-insensitive.
            (Component::Prefix(pa), Component::Prefix(pb)) => {
                pa.as_os_str().to_string_lossy().eq_ignore_ascii_case(&pb.as_os_str().to_string_lossy())
            }
            _ => a == b,
        }
    }
    #[cfg(not(windows))]
    {
        a == b
    }
}

/// Writes `value` as JSON atomically via a unique sibling temporary file and rename, cleaning up only the
/// temporary file created by this call if serialization, flushing, or renaming fails.
fn write_json_atomic<T: Serialize>(path: &Path, value: &T, pretty: bool) -> Result<()> {
    write_file_atomic(path, |writer| {
        let ser = if pretty {
            serde_json::to_writer_pretty(writer, value)
        } else {
            serde_json::to_writer(writer, value)
        };
        ser.map_err(|e| DataError::Format(format!("failed to serialize {}: {e}", path.display())))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_coco_names_prevent_traversal_and_collisions() {
        let paths =
            [Path::new("dir/a.jpg"), Path::new("other/A.png"), Path::new("x/.."), Path::new("y/a:b.jpg")];
        let mut taken = HashSet::new();
        let got: Vec<String> = paths
            .into_iter()
            .map(|p| {
                let (stem, ext) = sanitize_stem_and_ext(p);
                let mut n = 0usize;
                next_candidate_name(&stem, &ext, &mut taken, &mut n).unwrap()
            })
            .collect();
        assert_eq!(got, vec!["a.jpg", "A_1.png", "image.jpg", "a_b.jpg"]);
        for name in &got {
            assert!(!name.contains(['/', '\\', ':', '\0']) && !name.contains(".."), "{name}");
        }
    }

    #[test]
    fn diff_paths_and_lexical_normalize_handle_roots() {
        let base = Path::new("/tmp/work/coco-ref/annotations");
        let target = Path::new("/tmp/work/yolo/images/train/img_01.png");
        let rel = diff_paths(target, base).unwrap();
        assert_eq!(rel.to_string_lossy().replace('\\', "/"), "../../yolo/images/train/img_01.png");
        assert_eq!(lexical_normalize(Path::new("/../a/b/../c")), Path::new("/a/c"));
    }
}
