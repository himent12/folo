//! Dataset curation: class remapping, subset filtering, and multi-dataset merging into clean YOLO datasets.
//!
//! All operations work on opened [`LoadedDataset`]s (YOLO, COCO, or Pascal VOC) without re-encoding images:
//! - [`remap_classes`]: rename, unify, or drop classes with exact or case/whitespace-normalized matching.
//! - [`filter_dataset`] / [`write_filtered`]: carve clean subsets by class whitelist/blacklist, split selection,
//!   normalized box area, box/image pixel dimensions, and optional empty-image dropping.
//! - [`merge_datasets`] / [`write_merged`] / [`merge_paths`]: combine 2+ datasets (any mix of YOLO, COCO, VOC)
//!   with unified class names, explicit remapping/dropping, split preservation or re-splitting, and deterministic
//!   filename collision resolution.

use crate::bbox::BBox;
use crate::convert::{Converted, convert, copy, image_extension, mkdir};
use crate::dataset::{FileDataset, ImageEntry, LoadReport, LoadedDataset, open};
use crate::error::{DataError, Result};
use crate::image::probe_size;
use crate::split::{SplitConfig, stratified_indices};
use crate::{Format, Split};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn default_true() -> bool {
    true
}

/// How class names are matched and normalised during class remapping, subset filtering, or dataset merging.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassNormalize {
    /// Exact string equality (case- and whitespace-sensitive). Default.
    #[default]
    Exact,
    /// Trim outer whitespace, collapse internal whitespace runs to a single space, and compare lowercase.
    CaseAndWhitespace,
}

impl ClassNormalize {
    /// Canonical comparison key for a class name under this normalisation mode.
    pub fn key(self, name: &str) -> String {
        match self {
            ClassNormalize::Exact => name.to_string(),
            ClassNormalize::CaseAndWhitespace => collapse_ws(name).to_lowercase(),
        }
    }

    /// Cleaned display form of a class name (trimmed + single-space collapsed when `CaseAndWhitespace`).
    pub fn display(self, name: &str) -> String {
        match self {
            ClassNormalize::Exact => name.to_string(),
            ClassNormalize::CaseAndWhitespace => collapse_ws(name),
        }
    }
}

fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for (i, part) in s.split_whitespace().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(part);
    }
    out
}

/// Configuration for [`remap_classes`] / [`LoadedDataset::remap_classes`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemapConfig {
    /// Explicit mapping from source class name to target class name (`Some(new_name)`) or drop (`None`).
    /// Classes not listed keep their original name unless `drop_unmapped` is `true`.
    #[serde(default)]
    pub map: HashMap<String, Option<String>>,
    /// How source and target class names are matched and unified.
    #[serde(default)]
    pub normalize: ClassNormalize,
    /// When `true`, classes absent from `map` are dropped instead of kept as-is.
    #[serde(default)]
    pub drop_unmapped: bool,
    /// When `true`, remove images that end up with zero boxes after class dropping.
    #[serde(default)]
    pub drop_empty_images: bool,
}

/// Configuration for [`filter_dataset`] / [`LoadedDataset::filter`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Keep only boxes whose class name is in this whitelist (`None` keeps all classes).
    #[serde(default)]
    pub include_classes: Option<Vec<String>>,
    /// Drop boxes whose class name is in this blacklist (applied after `include_classes`).
    #[serde(default)]
    pub exclude_classes: Vec<String>,
    /// How class names in `include_classes` / `exclude_classes` are matched against `dataset.classes`.
    #[serde(default)]
    pub normalize: ClassNormalize,
    /// When `true` (default), compact `classes` to only the surviving classes and remap box class IDs to `0..n`.
    /// When `false`, keep the dataset's full original class list and IDs.
    #[serde(default = "default_true")]
    pub compact_classes: bool,
    /// Keep only these splits (`None` keeps all splits: `train`, `val`, `test`).
    #[serde(default)]
    pub splits: Option<Vec<Split>>,
    /// Minimum normalised box area `w * h` in `[0.0, 1.0]` (inclusive).
    #[serde(default)]
    pub min_box_area: Option<f32>,
    /// Maximum normalised box area `w * h` in `[0.0, 1.0]` (inclusive).
    #[serde(default)]
    pub max_box_area: Option<f32>,
    /// Minimum box width in pixels (inclusive).
    #[serde(default)]
    pub min_box_width_px: Option<f32>,
    /// Minimum box height in pixels (inclusive).
    #[serde(default)]
    pub min_box_height_px: Option<f32>,
    /// Maximum box width in pixels (inclusive).
    #[serde(default)]
    pub max_box_width_px: Option<f32>,
    /// Maximum box height in pixels (inclusive).
    #[serde(default)]
    pub max_box_height_px: Option<f32>,
    /// Minimum image width in pixels (inclusive).
    #[serde(default)]
    pub min_image_width: Option<u32>,
    /// Minimum image height in pixels (inclusive).
    #[serde(default)]
    pub min_image_height: Option<u32>,
    /// Maximum image width in pixels (inclusive).
    #[serde(default)]
    pub max_image_width: Option<u32>,
    /// Maximum image height in pixels (inclusive).
    #[serde(default)]
    pub max_image_height: Option<u32>,
    /// When `true`, remove images that have zero surviving boxes after filtering.
    #[serde(default)]
    pub drop_empty_images: bool,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            include_classes: None,
            exclude_classes: Vec::new(),
            normalize: ClassNormalize::Exact,
            compact_classes: true,
            splits: None,
            min_box_area: None,
            max_box_area: None,
            min_box_width_px: None,
            min_box_height_px: None,
            max_box_width_px: None,
            max_box_height_px: None,
            min_image_width: None,
            min_image_height: None,
            max_image_width: None,
            max_image_height: None,
            drop_empty_images: false,
        }
    }
}

impl FilterConfig {
    /// Validates numeric filter bounds.
    pub fn validate(&self) -> Result<()> {
        validate_opt_range("box area", self.min_box_area, self.max_box_area, 0.0, 1.0)?;
        validate_opt_range("box width (px)", self.min_box_width_px, self.max_box_width_px, 0.0, f32::MAX)?;
        validate_opt_range("box height (px)", self.min_box_height_px, self.max_box_height_px, 0.0, f32::MAX)?;
        if let (Some(lo), Some(hi)) = (self.min_image_width, self.max_image_width)
            && lo > hi
        {
            return Err(DataError::Config(format!("min_image_width ({lo}) exceeds max_image_width ({hi})")));
        }
        if let (Some(lo), Some(hi)) = (self.min_image_height, self.max_image_height)
            && lo > hi
        {
            return Err(DataError::Config(format!(
                "min_image_height ({lo}) exceeds max_image_height ({hi})"
            )));
        }
        Ok(())
    }

    fn needs_image_size(&self) -> bool {
        self.min_box_width_px.is_some()
            || self.min_box_height_px.is_some()
            || self.max_box_width_px.is_some()
            || self.max_box_height_px.is_some()
            || self.min_image_width.is_some()
            || self.min_image_height.is_some()
            || self.max_image_width.is_some()
            || self.max_image_height.is_some()
    }
}

fn validate_opt_range(label: &str, min: Option<f32>, max: Option<f32>, lo: f32, hi: f32) -> Result<()> {
    if let Some(v) = min
        && (!v.is_finite() || v < lo || v > hi)
    {
        return Err(DataError::Config(format!("min {label} must be finite and in [{lo}, {hi}], got {v}")));
    }
    if let Some(v) = max
        && (!v.is_finite() || v < lo || v > hi)
    {
        return Err(DataError::Config(format!("max {label} must be finite and in [{lo}, {hi}], got {v}")));
    }
    if let (Some(a), Some(b)) = (min, max)
        && a > b
    {
        return Err(DataError::Config(format!("min {label} ({a}) exceeds max {label} ({b})")));
    }
    Ok(())
}

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

/// Builds a deterministic lookup table from normalised source class key -> `Option<display_target>`.
fn build_normalized_map(
    map: &HashMap<String, Option<String>>,
    normalize: ClassNormalize,
) -> Result<HashMap<String, Option<String>>> {
    let mut pairs: Vec<(&String, &Option<String>)> = map.iter().collect();
    pairs.sort_by_key(|(a, _)| *a);

    let mut out: HashMap<String, Option<String>> = HashMap::with_capacity(pairs.len());
    for (src, dst) in pairs {
        let src_key = normalize.key(src);
        let resolved = match dst {
            Some(name) => {
                let disp = normalize.display(name);
                if disp.is_empty() {
                    return Err(DataError::Config(format!(
                        "remapped target class name for {src:?} cannot be empty"
                    )));
                }
                Some(disp)
            }
            None => None,
        };
        if let Some(prev) = out.get(&src_key) {
            let same = match (prev, &resolved) {
                (None, None) => true,
                (Some(a), Some(b)) => normalize.key(a) == normalize.key(b),
                _ => false,
            };
            if !same {
                return Err(DataError::Config(format!(
                    "conflicting class remap entries normalise to the same key {src_key:?}: {prev:?} vs {resolved:?}"
                )));
            }
        } else {
            out.insert(src_key, resolved);
        }
    }
    Ok(out)
}

type UnifiedClasses = (Vec<String>, Vec<Vec<Option<u32>>>);

/// Resolves a list of input datasets' classes into one unified class list plus per-dataset `old_id -> Option<new_id>`
/// lookup tables.
fn unify_classes(
    datasets: &[&LoadedDataset],
    normalize: ClassNormalize,
    class_map: &HashMap<String, Option<String>>,
    drop_unmapped: bool,
) -> Result<UnifiedClasses> {
    let norm_map = build_normalized_map(class_map, normalize)?;
    let mut unified: Vec<String> = Vec::new();
    let mut key_to_id: HashMap<String, u32> = HashMap::new();
    let mut luts: Vec<Vec<Option<u32>>> = Vec::with_capacity(datasets.len());

    for ds in datasets {
        let mut lut = Vec::with_capacity(ds.classes.len());
        for raw_name in &ds.classes {
            let src_key = normalize.key(raw_name);
            let target_opt = match norm_map.get(&src_key) {
                Some(mapped) => mapped.clone(),
                None if drop_unmapped => None,
                None => {
                    let disp = normalize.display(raw_name);
                    if disp.is_empty() { None } else { Some(disp) }
                }
            };
            let new_id = match target_opt {
                None => None,
                Some(disp) => {
                    let target_key = normalize.key(&disp);
                    let id = *key_to_id.entry(target_key).or_insert_with(|| {
                        let next = unified.len() as u32;
                        unified.push(disp);
                        next
                    });
                    Some(id)
                }
            };
            lut.push(new_id);
        }
        luts.push(lut);
    }

    Ok((unified, luts))
}

fn remap_entry_boxes(entry: &ImageEntry, lut: &[Option<u32>], dataset_label: &str) -> Result<Vec<BBox>> {
    let mut out = Vec::with_capacity(entry.boxes.len());
    for b in &entry.boxes {
        let idx = b.class as usize;
        let mapped = *lut.get(idx).ok_or_else(|| {
            DataError::Config(format!(
                "{dataset_label}: box class id {} is out of range for {} classes ({})",
                b.class,
                lut.len(),
                crate::paths::shown(&entry.image)
            ))
        })?;
        if let Some(new_class) = mapped {
            let mut nb = *b;
            nb.class = new_class;
            if !out.contains(&nb) {
                out.push(nb);
            }
        }
    }
    Ok(out)
}

/// Remaps, unifies, or drops classes on an opened [`LoadedDataset`], compacting surviving class IDs to `0..n`.
///
/// Images and labels on disk are not modified; pass the returned [`LoadedDataset`] to [`convert`] to write it out.
pub fn remap_classes(dataset: &LoadedDataset, config: &RemapConfig) -> Result<LoadedDataset> {
    let (classes, luts) = unify_classes(&[dataset], config.normalize, &config.map, config.drop_unmapped)?;
    let lut = &luts[0];

    let mut splits: [Vec<ImageEntry>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for split in Split::ALL {
        let entries = dataset.split(split).entries();
        let mut kept = Vec::with_capacity(entries.len());
        for e in entries {
            let boxes = remap_entry_boxes(e, lut, &crate::paths::shown(&dataset.root))?;
            if config.drop_empty_images && boxes.is_empty() {
                continue;
            }
            kept.push(ImageEntry { image: e.image.clone(), label: e.label.clone(), size: e.size, boxes });
        }
        splits[split as usize] = kept;
    }

    let [train, val, test] = splits;
    Ok(LoadedDataset {
        format: dataset.format,
        root: dataset.root.clone(),
        classes: classes.clone(),
        train: FileDataset::new(Split::Train, classes.clone(), train),
        val: FileDataset::new(Split::Val, classes.clone(), val),
        test: FileDataset::new(Split::Test, classes, test),
        report: LoadReport::default(),
        sources: dataset.sources.clone(),
    })
}

/// Filters an opened [`LoadedDataset`] into a clean subset by class whitelist/blacklist, split selection,
/// normalised box area, pixel box/image dimensions, and optional empty-image dropping.
///
/// When pixel-dimension bounds are configured and an entry has no cached image size (`entry.size == None`),
/// image headers are probed in parallel (respecting [`crate::guard`]).
pub fn filter_dataset(dataset: &LoadedDataset, config: &FilterConfig) -> Result<LoadedDataset> {
    config.validate()?;

    let include_keys: Option<HashSet<String>> =
        config.include_classes.as_ref().map(|list| list.iter().map(|n| config.normalize.key(n)).collect());
    let exclude_keys: HashSet<String> =
        config.exclude_classes.iter().map(|n| config.normalize.key(n)).collect();

    // Build old_id -> Option<new_id> and output class names.
    let mut out_classes: Vec<String> = Vec::new();
    let mut class_lut: Vec<Option<u32>> = Vec::with_capacity(dataset.classes.len());
    for (old_id, name) in dataset.classes.iter().enumerate() {
        let key = config.normalize.key(name);
        let included = include_keys.as_ref().is_none_or(|set| set.contains(&key));
        let excluded = exclude_keys.contains(&key);
        if included && !excluded {
            let new_id = if config.compact_classes {
                let id = out_classes.len() as u32;
                out_classes.push(name.clone());
                id
            } else {
                old_id as u32
            };
            class_lut.push(Some(new_id));
        } else {
            class_lut.push(None);
        }
    }
    if !config.compact_classes {
        out_classes = dataset.classes.clone();
    }

    let need_size = config.needs_image_size();
    let mut splits: [Vec<ImageEntry>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for split in Split::ALL {
        let keep_split = config.splits.as_ref().is_none_or(|allowed| allowed.contains(&split));
        if !keep_split {
            continue;
        }
        let entries = dataset.split(split).entries();
        let resolved_sizes: Vec<Option<(u32, u32)>> = if need_size {
            entries
                .par_iter()
                .map(|e| match e.size {
                    Some(sz) => Ok(Some(sz)),
                    None => probe_size(&e.image).map(Some),
                })
                .collect::<Result<_>>()?
        } else {
            entries.iter().map(|e| e.size).collect()
        };

        let out_of_bounds = |val: f32, min: Option<f32>, max: Option<f32>| -> bool {
            (min.is_some() || max.is_some())
                && (!val.is_finite() || min.is_some_and(|m| val < m) || max.is_some_and(|m| val > m))
        };

        let mut kept = Vec::with_capacity(entries.len());
        for (e, sz) in entries.iter().zip(resolved_sizes) {
            if let Some((iw, ih)) = sz
                && (config.min_image_width.is_some_and(|m| iw < m)
                    || config.max_image_width.is_some_and(|m| iw > m)
                    || config.min_image_height.is_some_and(|m| ih < m)
                    || config.max_image_height.is_some_and(|m| ih > m))
            {
                continue;
            }

            let mut boxes = Vec::with_capacity(e.boxes.len());
            for b in &e.boxes {
                let idx = b.class as usize;
                let mapped = *class_lut.get(idx).ok_or_else(|| {
                    DataError::Config(format!(
                        "box class id {} is out of range for {} classes ({})",
                        b.class,
                        class_lut.len(),
                        crate::paths::shown(&e.image)
                    ))
                })?;
                let Some(new_class) = mapped else {
                    continue;
                };
                if out_of_bounds(b.area(), config.min_box_area, config.max_box_area) {
                    continue;
                }
                if let Some((iw, ih)) = sz {
                    let bw = b.w * iw as f32;
                    let bh = b.h * ih as f32;
                    if out_of_bounds(bw, config.min_box_width_px, config.max_box_width_px)
                        || out_of_bounds(bh, config.min_box_height_px, config.max_box_height_px)
                    {
                        continue;
                    }
                }
                let mut nb = *b;
                nb.class = new_class;
                boxes.push(nb);
            }

            if config.drop_empty_images && boxes.is_empty() {
                continue;
            }
            kept.push(ImageEntry { image: e.image.clone(), label: e.label.clone(), size: sz, boxes });
        }
        splits[split as usize] = kept;
    }

    let [train, val, test] = splits;
    Ok(LoadedDataset {
        format: dataset.format,
        root: dataset.root.clone(),
        classes: out_classes.clone(),
        train: FileDataset::new(Split::Train, out_classes.clone(), train),
        val: FileDataset::new(Split::Val, out_classes.clone(), val),
        test: FileDataset::new(Split::Test, out_classes, test),
        report: LoadReport::default(),
        sources: dataset.sources.clone(),
    })
}

/// Filters `dataset` via [`filter_dataset`] and materialises the subset on disk in format `to` (`Yolo`, `Coco`,
/// or `Voc`) via [`convert`].
pub fn write_filtered(
    dataset: &LoadedDataset,
    out: &Path,
    to: Format,
    config: &FilterConfig,
) -> Result<Converted> {
    let filtered = filter_dataset(dataset, config)?;
    convert(&filtered, out, to)
}

struct MergedInternal {
    classes: Vec<String>,
    /// Per split `[train, val, test]`: `(source_dataset_index, entry)`.
    splits: [Vec<(usize, ImageEntry)>; 3],
    sources: Vec<PathBuf>,
    root: PathBuf,
}

fn merge_internal(datasets: &[&LoadedDataset], config: &MergeConfig) -> Result<MergedInternal> {
    if datasets.is_empty() {
        return Err(DataError::Config("merge requires at least one dataset".into()));
    }
    if !config.source_prefixes.is_empty() && config.source_prefixes.len() != datasets.len() {
        return Err(DataError::Config(format!(
            "source_prefixes length ({}) must match number of input datasets ({})",
            config.source_prefixes.len(),
            datasets.len()
        )));
    }
    if let MergeSplitPolicy::Stratified(split_cfg) = config.splits {
        split_cfg.validate()?;
    }

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

    Ok(MergedInternal { classes, splits, sources, root: datasets[0].root.clone() })
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

/// Combines `datasets` according to `config` and writes a clean YOLO dataset (`data.yaml` + `images/<split>/` +
/// `labels/<split>/`) to `out`, copying images byte-for-byte with deterministic collision-safe filenames.
///
/// `out` must not exist or must be an empty directory ([`crate::unfinished::claim`]). Fails with
/// [`DataError::Config`] before creating `out` when the merged dataset has no `train` or `val` images.
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

    let prefixes = resolve_prefixes(datasets.len(), &config.source_prefixes)?;
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

        let (names, split_renames) = resolve_merge_filenames(items, &prefixes, config.collisions);
        renamed_collisions += split_renames;

        items.par_iter().zip(&names).try_for_each(|((_, e), name)| copy(&e.image, &img_dir.join(name)))?;
        items.par_iter().zip(&names).try_for_each(|((_, e), name)| {
            let mut text = String::with_capacity(e.boxes.len() * 40);
            for b in &e.boxes {
                let _ = writeln!(text, "{} {:.6} {:.6} {:.6} {:.6}", b.class, b.cx, b.cy, b.w, b.h);
            }
            let p = lbl_dir.join(Path::new(name).with_extension("txt"));
            std::fs::write(&p, text).map_err(|err| DataError::io(&p, err))
        })?;

        let _ = writeln!(yaml_splits, "{split}: images/{split}");
        split_counts[split as usize] = items.len();
        boxes += items.iter().map(|(_, e)| e.boxes.len()).sum::<usize>();
    }

    let mut yaml = format!("# merged by FOLO from {} datasets\n{yaml_splits}names:\n", datasets.len());
    for (i, c) in merged.classes.iter().enumerate() {
        let _ = writeln!(yaml, "  {i}: {}", serde_json::to_string(c).expect("strings serialize"));
    }
    mkdir(out)?;
    let yaml_path = out.join("data.yaml");
    std::fs::write(&yaml_path, yaml).map_err(|e| DataError::io(&yaml_path, e))?;

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
    if inputs.is_empty() {
        return Err(DataError::Config("merge requires at least one dataset path".into()));
    }
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
        if cleaned.is_empty() || cleaned.chars().all(|c| c == '_') {
            return Err(DataError::Config(format!(
                "source_prefixes[{i}] ({raw:?}) must contain at least one alphanumeric character"
            )));
        }
        out.push(cleaned);
    }
    Ok(out)
}

/// Generates collision-free output filenames for one split of a merged dataset, comparing stems case-insensitively
/// so Windows and macOS filesystems (and YOLO `.txt` label pairing) never overwrite an entry.
fn resolve_merge_filenames(
    items: &[(usize, ImageEntry)],
    prefixes: &[String],
    policy: CollisionPolicy,
) -> (Vec<String>, usize) {
    // Count how often each lowercase stem appears across the split so colliding entries are prefixed cleanly.
    let mut stem_counts: HashMap<String, usize> = HashMap::with_capacity(items.len());
    let raw_stems: Vec<String> = items
        .iter()
        .map(|(_, e)| {
            let s = e
                .image
                .file_stem()
                .map(|x| x.to_string_lossy().into_owned())
                .unwrap_or_else(|| "image".into());
            *stem_counts.entry(s.to_lowercase()).or_insert(0) += 1;
            s
        })
        .collect();

    let mut taken: HashSet<String> = HashSet::with_capacity(items.len());
    let mut names: Vec<String> = Vec::with_capacity(items.len());
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
        names.push(format!("{candidate}{ext}"));
    }

    (names, renamed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_normalize_and_conflict_detection() {
        assert_eq!(ClassNormalize::Exact.key(" Person "), " Person ");
        assert_eq!(ClassNormalize::CaseAndWhitespace.key("  Traffic   Light \t"), "traffic light");
        assert_eq!(ClassNormalize::CaseAndWhitespace.display("  Traffic   Light "), "Traffic Light");

        // Conflicting case variants under CaseAndWhitespace must error cleanly.
        let mut map = HashMap::new();
        map.insert("Car".into(), Some("vehicle".into()));
        map.insert("car".into(), Some("auto".into()));
        assert!(build_normalized_map(&map, ClassNormalize::CaseAndWhitespace).is_err());

        // Agreeing case variants under CaseAndWhitespace succeed deterministically.
        let mut ok_map = HashMap::new();
        ok_map.insert("Car".into(), Some("Vehicle".into()));
        ok_map.insert("car".into(), Some("vehicle".into()));
        let built = build_normalized_map(&ok_map, ClassNormalize::CaseAndWhitespace).unwrap();
        assert_eq!(built.get("car"), Some(&Some("Vehicle".into())));
    }

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

        let (names, renamed) = resolve_merge_filenames(&items, &prefixes, CollisionPolicy::PrefixOnCollision);
        assert_eq!(
            names,
            vec!["ds0_0001.jpg", "unique.png", "ds1_0001.png", "ds1_0001_1.JPG", "ds1_0001_2.jpg"]
        );
        assert_eq!(renamed, 4);

        let (prefixed, renamed_all) =
            resolve_merge_filenames(&items, &prefixes, CollisionPolicy::AlwaysPrefix);
        assert_eq!(
            prefixed,
            vec!["ds0_0001.jpg", "ds0_unique.png", "ds1_0001.png", "ds1_0001_1.JPG", "ds0_ds1_0001.jpg"]
        );
        assert_eq!(renamed_all, 5);
    }

    #[test]
    fn filter_config_validates_bounds() {
        assert!(FilterConfig::default().validate().is_ok());
        assert!(FilterConfig { min_box_area: Some(-0.1), ..Default::default() }.validate().is_err());
        assert!(FilterConfig { max_box_area: Some(1.5), ..Default::default() }.validate().is_err());
        assert!(
            FilterConfig { min_box_area: Some(0.5), max_box_area: Some(0.2), ..Default::default() }
                .validate()
                .is_err()
        );
        assert!(FilterConfig { min_box_width_px: Some(f32::NAN), ..Default::default() }.validate().is_err());
        assert!(
            FilterConfig { min_image_width: Some(640), max_image_width: Some(320), ..Default::default() }
                .validate()
                .is_err()
        );
    }
}
