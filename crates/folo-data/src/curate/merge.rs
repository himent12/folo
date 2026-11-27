//! Multi-dataset merging (`merge_datasets`, `write_merged`, `merge_paths`) with class unification,
//! split policies, and deterministic collision-safe YOLO materialisation.

use super::{ClassNormalize, build_normalized_map, remap_entry_boxes, unify_classes};
use crate::bbox::BBox;
use crate::convert::{image_extension, mkdir};
use crate::dataset::{FileDataset, ImageEntry, LoadReport, LoadedDataset, open};
use crate::error::{DataError, Result};
use crate::split::{SplitConfig, stratified_indices};
use crate::{Format, Split};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// How splits from input datasets are combined during [`merge_datasets`] / [`write_merged`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MergeSplitPolicy {
    /// Keep each image in its original split (`train` -> `train`, `val` -> `val`, `test` -> `test`). Default.
    #[default]
    Preserve,
    /// Pool all images from every input split into `train` (leaving `val` and `test` empty).
    PoolToTrain,
    /// Pool all images from every input dataset and re-split them via deterministic multi-label stratification.
    Stratified(SplitConfig),
}

/// How output image filenames are disambiguated when writing a merged dataset to disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollisionPolicy {
    /// Keep original filenames when stems are unique within a split; disambiguate any collision
    /// deterministically by prefixing colliding entries with their source tag (`<prefix>_<stem>.<ext>`,
    /// appending `_N` if a source itself contains colliding stems). Default.
    #[default]
    PrefixOnCollision,
    /// Prefix every image filename with its source tag (`<prefix>_<stem>.<ext>`), appending `_N` for
    /// intra-source stem collisions.
    AlwaysPrefix,
}

/// Configuration for [`merge_datasets`], [`write_merged`], and [`merge_paths`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MergeConfig {
    /// How class names across datasets are matched and unified (`Exact` or `CaseAndWhitespace`).
    #[serde(default)]
    pub normalize: ClassNormalize,
    /// Optional explicit class remap/drop table (`source_name -> Some(target_name)` or `None` to drop).
    /// Applied to every input dataset before unifying classes.
    #[serde(default)]
    pub class_map: HashMap<String, Option<String>>,
    /// When `true`, classes absent from `class_map` are dropped instead of kept.
    #[serde(default)]
    pub drop_unmapped: bool,
    /// When `true`, drop images that end up with zero boxes after class dropping.
    #[serde(default)]
    pub drop_empty_images: bool,
    /// How splits from input datasets are combined.
    #[serde(default)]
    pub splits: MergeSplitPolicy,
    /// How filename collisions are resolved when materialising on disk.
    #[serde(default)]
    pub collisions: CollisionPolicy,
    /// Optional per-dataset filename prefixes used when disambiguating collisions (must match the number of
    /// input datasets when non-empty; defaults to `ds0`, `ds1`, ...).
    #[serde(default)]
    pub source_prefixes: Vec<String>,
}

/// Summary of a merged YOLO dataset written to disk by [`write_merged`] / [`merge_paths`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergeReport {
    /// Output root directory.
    pub root: PathBuf,
    /// Unified class names written to `data.yaml`.
    pub classes: Vec<String>,
    /// Images written per split `[train, val, test]`.
    pub splits: [usize; 3],
    /// Total boxes written across all splits.
    pub boxes: usize,
    /// Number of output image filenames that were prefixed or suffixed (due to collisions or
    /// [`CollisionPolicy::AlwaysPrefix`]).
    pub renamed_collisions: usize,
}

struct MergedInternal {
    classes: Vec<String>,
    /// Per split `[train, val, test]`: `(source_dataset_index, entry)`.
    splits: [Vec<(usize, ImageEntry)>; 3],
    sources: Vec<PathBuf>,
    root: PathBuf,
    prefixes: Vec<String>,
}

/// Validates `config` for `dataset_count` inputs up front (before opening files or claiming output directories).
fn validate_merge_config(config: &MergeConfig, dataset_count: usize) -> Result<Vec<String>> {
    if dataset_count == 0 {
        return Err(DataError::Config("merge requires at least one dataset".into()));
    }
    if !config.source_prefixes.is_empty() && config.source_prefixes.len() != dataset_count {
        return Err(DataError::Config(format!(
            "source_prefixes length ({}) must match number of input datasets ({dataset_count})",
            config.source_prefixes.len(),
        )));
    }
    if let MergeSplitPolicy::Stratified(split_cfg) = config.splits {
        split_cfg.validate()?;
    }
    let _ = build_normalized_map(&config.class_map, config.normalize)?;
    resolve_prefixes(dataset_count, &config.source_prefixes)
}

fn merge_internal(datasets: &[&LoadedDataset], config: &MergeConfig) -> Result<MergedInternal> {
    let prefixes = validate_merge_config(config, datasets.len())?;
    let (classes, luts) = unify_classes(datasets, config.normalize, &config.class_map, config.drop_unmapped)?;

    let mut sources: Vec<PathBuf> = Vec::new();
    let mut seen_sources: HashSet<PathBuf> = HashSet::new();
    for ds in datasets {
        for s in &ds.sources {
            if seen_sources.insert(s.clone()) {
                sources.push(s.clone());
            }
        }
    }

    let mut splits: [Vec<(usize, ImageEntry)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    match config.splits {
        MergeSplitPolicy::Preserve => {
            for (ds_idx, (ds, lut)) in datasets.iter().zip(&luts).enumerate() {
                let label = format!("dataset #{ds_idx} ({})", crate::paths::shown(&ds.root));
                for split in Split::ALL {
                    for e in ds.split(split).entries() {
                        let boxes = remap_entry_boxes(e, lut, &label)?;
                        if config.drop_empty_images && boxes.is_empty() {
                            continue;
                        }
                        splits[split as usize].push((
                            ds_idx,
                            ImageEntry {
                                image: e.image.clone(),
                                label: e.label.clone(),
                                size: e.size,
                                boxes,
                            },
                        ));
                    }
                }
            }
        }
        MergeSplitPolicy::PoolToTrain => {
            for (ds_idx, (ds, lut)) in datasets.iter().zip(&luts).enumerate() {
                let label = format!("dataset #{ds_idx} ({})", crate::paths::shown(&ds.root));
                for split in Split::ALL {
                    for e in ds.split(split).entries() {
                        let boxes = remap_entry_boxes(e, lut, &label)?;
                        if config.drop_empty_images && boxes.is_empty() {
                            continue;
                        }
                        splits[Split::Train as usize].push((
                            ds_idx,
                            ImageEntry {
                                image: e.image.clone(),
                                label: e.label.clone(),
                                size: e.size,
                                boxes,
                            },
                        ));
                    }
                }
            }
        }
        MergeSplitPolicy::Stratified(split_cfg) => {
            let mut pooled: Vec<(usize, ImageEntry)> = Vec::new();
            for (ds_idx, (ds, lut)) in datasets.iter().zip(&luts).enumerate() {
                let label = format!("dataset #{ds_idx} ({})", crate::paths::shown(&ds.root));
                for split in Split::ALL {
                    for e in ds.split(split).entries() {
                        let boxes = remap_entry_boxes(e, lut, &label)?;
                        if config.drop_empty_images && boxes.is_empty() {
                            continue;
                        }
                        pooled.push((
                            ds_idx,
                            ImageEntry {
                                image: e.image.clone(),
                                label: e.label.clone(),
                                size: e.size,
                                boxes,
                            },
                        ));
                    }
                }
            }
            let labels: Vec<Vec<u32>> =
                pooled.iter().map(|(_, e)| e.boxes.iter().map(|b| b.class).collect()).collect();
            let idxs = stratified_indices(&labels, &split_cfg)?;
            for (split, list) in
                [(Split::Train, &idxs.train), (Split::Val, &idxs.val), (Split::Test, &idxs.test)]
            {
                splits[split as usize] = list.iter().map(|&i| pooled[i].clone()).collect();
            }
        }
    }

    Ok(MergedInternal { classes, splits, sources, root: datasets[0].root.clone(), prefixes })
}

/// Validates that every box can be serialised to YOLO text and read back cleanly without losing annotations.
fn validate_yolo_boxes(entry: &ImageEntry, num_classes: usize) -> Result<()> {
    for b in &entry.boxes {
        if (b.class as usize) >= num_classes {
            return Err(DataError::Config(format!(
                "box class id {} is out of range for {num_classes} classes ({})",
                b.class,
                crate::paths::shown(&entry.image)
            )));
        }
        if !b.cx.is_finite() || !b.cy.is_finite() || !b.w.is_finite() || !b.h.is_finite() {
            return Err(DataError::Config(format!(
                "non-finite box coordinates in {}",
                crate::paths::shown(&entry.image)
            )));
        }
        if b.w <= 0.0 || b.h <= 0.0 {
            return Err(DataError::Config(format!(
                "non-positive box size ({}x{}) in {}",
                b.w,
                b.h,
                crate::paths::shown(&entry.image)
            )));
        }
        let [x1, y1, x2, y2] = b.xyxy();
        if x1 < -0.01 || y1 < -0.01 || x2 > 1.01 || y2 > 1.01 {
            return Err(DataError::Config(format!(
                "out-of-range normalised box [{x1}, {y1}, {x2}, {y2}] in {}",
                crate::paths::shown(&entry.image)
            )));
        }
    }
    Ok(())
}

/// Formats boxes with exact shortest round-trip `f32` decimals (matching [`crate::pack`]) so small valid boxes
/// are never rounded down to `0.000000` and dropped on reopen.
fn format_yolo_labels(boxes: &[BBox]) -> String {
    let mut text = String::with_capacity(boxes.len() * 48);
    for b in boxes {
        let _ = writeln!(text, "{} {} {} {} {}", b.class, b.cx, b.cy, b.w, b.h);
    }
    text
}

/// Copies `from` to `to` using `create_new(true)` so a filesystem alias can never silently overwrite another file.
fn copy_exclusive(from: &Path, to: &Path) -> Result<()> {
    crate::guard::check(from).map_err(|e| DataError::io(from, e))?;
    let mut src = std::fs::File::open(from).map_err(|e| DataError::io(from, e))?;
    let mut dst =
        OpenOptions::new().write(true).create_new(true).open(to).map_err(|e| DataError::io(to, e))?;
    std::io::copy(&mut src, &mut dst).map_err(|e| DataError::io(to, e))?;
    Ok(())
}

/// Writes `content` to `to` using `create_new(true)` so a filesystem alias can never silently overwrite another file.
fn write_exclusive(to: &Path, content: &[u8]) -> Result<()> {
    let mut dst =
        OpenOptions::new().write(true).create_new(true).open(to).map_err(|e| DataError::io(to, e))?;
    dst.write_all(content).map_err(|e| DataError::io(to, e))
}

/// Writes one entry sequentially with exclusive creation, disambiguating OS-level filename aliases (such as
/// Unicode NFC/NFD normalization on APFS) by bumping `_N` if either the image or label file already exists.
fn write_entry_resolving_os_aliases(
    entry: &ImageEntry,
    initial_stem: &str,
    ext: &str,
    img_dir: &Path,
    lbl_dir: &Path,
) -> Result<bool> {
    crate::guard::check(&entry.image).map_err(|e| DataError::io(&entry.image, e))?;
    let label_text = format_yolo_labels(&entry.boxes);
    let mut candidate = initial_stem.to_string();
    let mut n = 0usize;
    loop {
        let lbl_path = lbl_dir.join(format!("{candidate}.txt"));
        match OpenOptions::new().write(true).create_new(true).open(&lbl_path) {
            Ok(mut lbl_file) => {
                let img_path = img_dir.join(format!("{candidate}{ext}"));
                match OpenOptions::new().write(true).create_new(true).open(&img_path) {
                    Ok(mut img_file) => {
                        let mut src =
                            std::fs::File::open(&entry.image).map_err(|e| DataError::io(&entry.image, e))?;
                        std::io::copy(&mut src, &mut img_file).map_err(|e| DataError::io(&img_path, e))?;
                        lbl_file.write_all(label_text.as_bytes()).map_err(|e| DataError::io(&lbl_path, e))?;
                        return Ok(n > 0);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                        drop(lbl_file);
                        let _ = std::fs::remove_file(&lbl_path);
                        n += 1;
                        candidate = format!("{initial_stem}_{n}");
                    }
                    Err(e) => {
                        drop(lbl_file);
                        let _ = std::fs::remove_file(&lbl_path);
                        return Err(DataError::io(&img_path, e));
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                n += 1;
                candidate = format!("{initial_stem}_{n}");
            }
            Err(e) => return Err(DataError::io(&lbl_path, e)),
        }
    }
}

/// Combines one or more opened [`LoadedDataset`]s (any mix of YOLO, COCO, and Pascal VOC) in memory with unified
/// class names, optional remapping/dropping, and configurable split handling.
pub fn merge_datasets(datasets: &[&LoadedDataset], config: &MergeConfig) -> Result<LoadedDataset> {
    let merged = merge_internal(datasets, config)?;
    let [train, val, test] = merged.splits;
    let take = |split: Split, items: Vec<(usize, ImageEntry)>| {
        let entries: Vec<ImageEntry> = items.into_iter().map(|(_, e)| e).collect();
        FileDataset::new(split, merged.classes.clone(), entries)
    };
    Ok(LoadedDataset {
        format: Format::Yolo,
        root: merged.root,
        classes: merged.classes.clone(),
        train: take(Split::Train, train),
        val: take(Split::Val, val),
        test: take(Split::Test, test),
        report: LoadReport::default(),
        sources: merged.sources,
    })
}

/// Combines one or more `datasets` according to `config` and writes a clean YOLO dataset (`data.yaml` +
/// `images/<split>/` + `labels/<split>/`) to `out`, copying images byte-for-byte with deterministic
/// collision-safe filenames and exclusive file creation.
///
/// `out` must not exist or must be an empty directory ([`crate::unfinished::claim`]). Fails with
/// [`DataError::Config`] before creating `out` when the merged dataset has no `train` or `val` images or
/// contains invalid box coordinates.
pub fn write_merged(datasets: &[&LoadedDataset], out: &Path, config: &MergeConfig) -> Result<MergeReport> {
    let merged = merge_internal(datasets, config)?;
    let train_len = merged.splits[Split::Train as usize].len();
    let val_len = merged.splits[Split::Val as usize].len();
    let test_len = merged.splits[Split::Test as usize].len();
    if train_len + val_len + test_len == 0 {
        return Err(DataError::Config("nothing to merge: combined dataset has no readable images".into()));
    }
    if train_len + val_len == 0 {
        return Err(DataError::Config(
            "a YOLO dataset needs train or val images, and the merged dataset only has test images".into(),
        ));
    }
    for split in Split::ALL {
        for (_, e) in &merged.splits[split as usize] {
            validate_yolo_boxes(e, merged.classes.len())?;
        }
    }

    let claim = crate::unfinished::claim(out)?;

    let mut split_counts = [0usize; 3];
    let mut boxes = 0usize;
    let mut renamed_collisions = 0usize;
    let mut yaml_splits = String::new();

    for split in Split::ALL {
        let items = &merged.splits[split as usize];
        if items.is_empty() {
            continue;
        }
        let img_dir = out.join("images").join(split.as_str());
        let lbl_dir = out.join("labels").join(split.as_str());
        mkdir(&img_dir)?;
        mkdir(&lbl_dir)?;

        let (resolved, split_renames) = resolve_merge_filenames(items, &merged.prefixes, config.collisions);
        renamed_collisions += split_renames;

        let has_non_ascii = resolved.iter().any(|(stem, _)| !stem.is_ascii());
        if has_non_ascii {
            // Sequential exclusive creation catches OS-level Unicode normalization aliases (e.g. APFS NFC/NFD).
            for ((_, e), (stem, ext)) in items.iter().zip(&resolved) {
                if write_entry_resolving_os_aliases(e, stem, ext, &img_dir, &lbl_dir)? {
                    renamed_collisions += 1;
                }
            }
        } else {
            items.par_iter().zip(&resolved).try_for_each(|((_, e), (stem, ext))| {
                copy_exclusive(&e.image, &img_dir.join(format!("{stem}{ext}")))
            })?;
            items.par_iter().zip(&resolved).try_for_each(|((_, e), (stem, _))| {
                let text = format_yolo_labels(&e.boxes);
                write_exclusive(&lbl_dir.join(format!("{stem}.txt")), text.as_bytes())
            })?;
        }

        let _ = writeln!(yaml_splits, "{split}: images/{split}");
        split_counts[split as usize] = items.len();
        boxes += items.iter().map(|(_, e)| e.boxes.len()).sum::<usize>();
    }

    let mut yaml = format!("# merged by FOLO from {} datasets\n{yaml_splits}names:\n", datasets.len());
    for (i, c) in merged.classes.iter().enumerate() {
        let _ = writeln!(yaml, "  {i}: {}", serde_json::to_string(c).expect("strings serialize"));
    }
    mkdir(out)?;
    write_exclusive(&out.join("data.yaml"), yaml.as_bytes())?;

    claim.finish()?;
    Ok(MergeReport {
        root: out.to_path_buf(),
        classes: merged.classes,
        splits: split_counts,
        boxes,
        renamed_collisions,
    })
}

/// Opens every input dataset path (auto-detecting YOLO, COCO, or Pascal VOC), merges them according to `config`,
/// and writes a clean YOLO dataset to `out`.
pub fn merge_paths(inputs: &[impl AsRef<Path>], out: &Path, config: &MergeConfig) -> Result<MergeReport> {
    // Validate configuration up front before opening and probing datasets on disk.
    let _ = validate_merge_config(config, inputs.len())?;
    let opened: Vec<LoadedDataset> = inputs.iter().map(open).collect::<Result<_>>()?;
    let refs: Vec<&LoadedDataset> = opened.iter().collect();
    write_merged(&refs, out, config)
}

fn sanitize_prefix(raw: &str) -> String {
    raw.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') { c } else { '_' }).collect()
}

fn resolve_prefixes(count: usize, custom: &[String]) -> Result<Vec<String>> {
    if custom.is_empty() {
        return Ok((0..count).map(|i| format!("ds{i}")).collect());
    }
    let mut out = Vec::with_capacity(count);
    for (i, raw) in custom.iter().enumerate() {
        let cleaned = sanitize_prefix(raw.trim());
        if !cleaned.chars().any(|c| c.is_ascii_alphanumeric()) {
            return Err(DataError::Config(format!(
                "source_prefixes[{i}] ({raw:?}) must contain at least one ASCII alphanumeric character"
            )));
        }
        out.push(cleaned);
    }
    Ok(out)
}

/// Normalises a filename stem for cross-platform safety (trimming Windows-aliased trailing dots/spaces).
fn clean_stem(p: &Path) -> String {
    let raw = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "image".into());
    let trimmed = raw.trim_end_matches(['.', ' ']);
    if trimmed.is_empty() { "image".to_string() } else { trimmed.to_string() }
}

/// Generates collision-free `(stem, ".ext")` pairs for one split of a merged dataset, comparing stems
/// case-insensitively so Windows and macOS filesystems (and YOLO `.txt` label pairing) never overwrite an entry.
fn resolve_merge_filenames(
    items: &[(usize, ImageEntry)],
    prefixes: &[String],
    policy: CollisionPolicy,
) -> (Vec<(String, String)>, usize) {
    let mut stem_counts: HashMap<String, usize> = HashMap::with_capacity(items.len());
    let raw_stems: Vec<String> = items
        .iter()
        .map(|(_, e)| {
            let s = clean_stem(&e.image);
            *stem_counts.entry(s.to_lowercase()).or_insert(0) += 1;
            s
        })
        .collect();

    let mut taken: HashSet<String> = HashSet::with_capacity(items.len());
    let mut names: Vec<(String, String)> = Vec::with_capacity(items.len());
    let mut renamed = 0usize;

    for ((ds_idx, e), stem) in items.iter().zip(raw_stems) {
        let ext = format!(".{}", image_extension(&e.image));
        let has_collision = stem_counts.get(&stem.to_lowercase()).copied().unwrap_or(0) > 1;
        let base = match policy {
            CollisionPolicy::AlwaysPrefix => format!("{}_{stem}", prefixes[*ds_idx]),
            CollisionPolicy::PrefixOnCollision if has_collision => format!("{}_{stem}", prefixes[*ds_idx]),
            CollisionPolicy::PrefixOnCollision => stem.clone(),
        };
        let mut candidate = base.clone();
        let mut n = 0usize;
        while !taken.insert(candidate.to_lowercase()) {
            n += 1;
            candidate = format!("{base}_{n}");
        }
        if candidate != stem {
            renamed += 1;
        }
        names.push((candidate, ext));
    }

    (names, renamed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_filename_collision_resolution_is_deterministic() {
        let make = |p: &str| ImageEntry { image: PathBuf::from(p), label: None, size: None, boxes: vec![] };
        let items = vec![
            (0, make("a/0001.jpg")),
            (0, make("a/unique.png")),
            (1, make("b/0001.png")),
            (1, make("b/0001.JPG")),
            (0, make("a/ds1_0001.jpg")),
        ];
        let prefixes = vec!["ds0".into(), "ds1".into()];

        let (pairs, renamed) = resolve_merge_filenames(&items, &prefixes, CollisionPolicy::PrefixOnCollision);
        let names: Vec<String> = pairs.into_iter().map(|(s, e)| format!("{s}{e}")).collect();
        assert_eq!(
            names,
            vec!["ds0_0001.jpg", "unique.png", "ds1_0001.png", "ds1_0001_1.JPG", "ds1_0001_2.jpg"]
        );
        assert_eq!(renamed, 4);

        let (pairs_all, renamed_all) =
            resolve_merge_filenames(&items, &prefixes, CollisionPolicy::AlwaysPrefix);
        let prefixed: Vec<String> = pairs_all.into_iter().map(|(s, e)| format!("{s}{e}")).collect();
        assert_eq!(
            prefixed,
            vec!["ds0_0001.jpg", "ds0_unique.png", "ds1_0001.png", "ds1_0001_1.JPG", "ds0_ds1_0001.jpg"]
        );
        assert_eq!(renamed_all, 5);
    }

    #[test]
    fn resolve_prefixes_rejects_hyphen_and_punctuation_only() {
        assert!(resolve_prefixes(2, &["---".into(), "ok".into()]).is_err());
        assert!(resolve_prefixes(2, &["-_-".into(), "ok".into()]).is_err());
        assert!(resolve_prefixes(1, &["   ".into()]).is_err());
        assert_eq!(
            resolve_prefixes(2, &["coco-2017".into(), "voc_12".into()]).unwrap(),
            ["coco-2017", "voc_12"]
        );
    }
}
