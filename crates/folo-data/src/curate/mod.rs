//! Dataset curation: class remapping, subset filtering, and multi-dataset merging into clean YOLO datasets.
//!
//! All operations work on opened [`LoadedDataset`]s (YOLO, COCO, or Pascal VOC) without re-encoding images:
//! - [`remap_classes`]: rename, unify, or drop classes with exact or case/whitespace-normalised matching.
//! - [`filter_dataset`] / [`write_filtered`]: carve clean subsets by class whitelist/blacklist, split selection,
//!   normalised box area, box/image pixel dimensions, and optional empty-image dropping.
//! - [`merge_datasets`] / [`write_merged`] / [`merge_paths`] (in `merge.rs`): combine one or more datasets (any
//!   mix of YOLO, COCO, VOC) with unified class names, explicit remapping/dropping, split preservation or
//!   re-splitting, and deterministic filename collision resolution.

mod merge;

pub use merge::{
    CollisionPolicy, MergeConfig, MergeReport, MergeSplitPolicy, merge_datasets, merge_paths, write_merged,
};

use crate::bbox::BBox;
use crate::convert::{Converted, convert};
use crate::dataset::labels::{BoxKey, DEDUP_HASH_THRESHOLD, box_key};
use crate::dataset::{FileDataset, ImageEntry, LoadReport, LoadedDataset};
use crate::error::{DataError, Result};
use crate::image::probe_size;
use crate::{Format, Split};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

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
    /// When `true` (default), compact `classes` to only the classes retained by `include_classes` /
    /// `exclude_classes` and remap box class IDs to `0..n` (classes kept by the class filter remain in
    /// `classes` even if zero boxes survive area/pixel/split bounds). When `false`, keep the dataset's full
    /// original class list and IDs.
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
    /// When `true`, always probe actual on-disk image headers for candidates that need pixel dimensions
    /// (instead of trusting `entry.size`, which may come from unverified COCO/VOC metadata when opened with
    /// `verify_images: false`). Default `false` (uses cached `entry.size` when present and lazily probes when
    /// `None`).
    #[serde(default)]
    pub probe_image_sizes: bool,
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
            probe_image_sizes: false,
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

    fn has_image_dim_bounds(&self) -> bool {
        self.min_image_width.is_some()
            || self.min_image_height.is_some()
            || self.max_image_width.is_some()
            || self.max_image_height.is_some()
    }

    fn has_box_px_bounds(&self) -> bool {
        self.min_box_width_px.is_some()
            || self.min_box_height_px.is_some()
            || self.max_box_width_px.is_some()
            || self.max_box_height_px.is_some()
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

/// Builds a deterministic lookup table from normalised source class key -> `Option<display_target>`.
pub(super) fn build_normalized_map(
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

pub(super) type UnifiedClasses = (Vec<String>, Vec<Vec<Option<u32>>>);

/// Resolves a list of input datasets' classes into one unified class list plus per-dataset `old_id -> Option<new_id>`
/// lookup tables. Rejects unmapped empty/whitespace-only class names rather than silently dropping their boxes.
pub(super) fn unify_classes(
    datasets: &[&LoadedDataset],
    normalize: ClassNormalize,
    class_map: &HashMap<String, Option<String>>,
    drop_unmapped: bool,
) -> Result<UnifiedClasses> {
    let norm_map = build_normalized_map(class_map, normalize)?;
    let mut unified: Vec<String> = Vec::new();
    let mut key_to_id: HashMap<String, u32> = HashMap::new();
    let mut luts: Vec<Vec<Option<u32>>> = Vec::with_capacity(datasets.len());

    for (ds_idx, ds) in datasets.iter().enumerate() {
        let mut lut = Vec::with_capacity(ds.classes.len());
        for (cls_idx, raw_name) in ds.classes.iter().enumerate() {
            let src_key = normalize.key(raw_name);
            let target_opt = match norm_map.get(&src_key) {
                Some(mapped) => mapped.clone(),
                None if drop_unmapped => None,
                None => {
                    let disp = normalize.display(raw_name);
                    if disp.is_empty() {
                        return Err(DataError::Config(format!(
                            "dataset #{ds_idx} class #{cls_idx} ({raw_name:?}) normalises to an empty class name; \
                             remap or drop it explicitly via class_map"
                        )));
                    }
                    Some(disp)
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

/// Remaps an entry's boxes using `lut` and deduplicates exact duplicate boxes in O(B) average time (promoting
/// to a hash set above [`DEDUP_HASH_THRESHOLD`], matching the YOLO label reader).
pub(super) fn remap_entry_boxes(
    entry: &ImageEntry,
    lut: &[Option<u32>],
    dataset_label: &str,
) -> Result<Vec<BBox>> {
    let mut out = Vec::with_capacity(entry.boxes.len());
    let mut seen: Option<HashSet<BoxKey>> = None;
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
        let Some(new_class) = mapped else {
            continue;
        };
        let mut nb = *b;
        nb.class = new_class;
        let is_dup =
            if let Some(set) = seen.as_mut() { !set.insert(box_key(&nb)) } else { out.contains(&nb) };
        if !is_dup {
            out.push(nb);
            if seen.is_none() && out.len() >= DEDUP_HASH_THRESHOLD {
                let mut set = HashSet::with_capacity(entry.boxes.len().max(DEDUP_HASH_THRESHOLD * 2));
                for kept in &out {
                    set.insert(box_key(kept));
                }
                seen = Some(set);
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
/// Image headers are probed lazily in parallel (respecting [`crate::guard`]) only for candidate images that
/// require pixel dimensions (`entry.size == None` or [`FilterConfig::probe_image_sizes`] is `true`). When only
/// box pixel bounds are set, images with no boxes surviving class and area filtering are not probed.
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

    let has_area_bounds = config.min_box_area.is_some() || config.max_box_area.is_some();
    let has_img_bounds = config.has_image_dim_bounds();
    let has_px_bounds = config.has_box_px_bounds();
    let out_of_bounds = |val: f32, min: Option<f32>, max: Option<f32>| -> bool {
        (min.is_some() || max.is_some())
            && (!val.is_finite() || val < 0.0 || min.is_some_and(|m| val < m) || max.is_some_and(|m| val > m))
    };

    let mut splits: [Vec<ImageEntry>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    for split in Split::ALL {
        let keep_split = config.splits.as_ref().is_none_or(|allowed| allowed.contains(&split));
        if !keep_split {
            continue;
        }
        let entries = dataset.split(split).entries();

        // Phase 1: Pure in-memory class & normalised-area filtering.
        let mut prefiltered: Vec<(usize, Vec<BBox>)> = Vec::with_capacity(entries.len());
        for (idx, e) in entries.iter().enumerate() {
            let mut boxes = Vec::with_capacity(e.boxes.len());
            for b in &e.boxes {
                let ci = b.class as usize;
                let mapped = *class_lut.get(ci).ok_or_else(|| {
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
                if has_area_bounds
                    && (!b.w.is_finite()
                        || !b.h.is_finite()
                        || b.w < 0.0
                        || b.h < 0.0
                        || out_of_bounds(b.area(), config.min_box_area, config.max_box_area))
                {
                    continue;
                }
                let mut nb = *b;
                nb.class = new_class;
                boxes.push(nb);
            }
            if config.drop_empty_images && boxes.is_empty() {
                continue;
            }
            prefiltered.push((idx, boxes));
        }

        // Phase 2: Resolve pixel sizes lazily only for entries that actually need them.
        let resolved_sizes: Vec<Option<(u32, u32)>> = if has_img_bounds || has_px_bounds {
            prefiltered
                .par_iter()
                .map(|(idx, boxes)| {
                    let e = &entries[*idx];
                    let entry_needs_size = has_img_bounds || (has_px_bounds && !boxes.is_empty());
                    if !entry_needs_size {
                        return Ok(e.size);
                    }
                    if config.probe_image_sizes {
                        return probe_size(&e.image).map(Some);
                    }
                    match e.size {
                        Some(sz) => Ok(Some(sz)),
                        None => probe_size(&e.image).map(Some),
                    }
                })
                .collect::<Result<_>>()?
        } else {
            prefiltered.iter().map(|(idx, _)| entries[*idx].size).collect()
        };

        // Phase 3: Apply image-dimension and pixel-box bounds.
        let mut kept = Vec::with_capacity(prefiltered.len());
        for ((idx, mut boxes), sz) in prefiltered.into_iter().zip(resolved_sizes) {
            let e = &entries[idx];
            if has_img_bounds
                && let Some((iw, ih)) = sz
                && (config.min_image_width.is_some_and(|m| iw < m)
                    || config.max_image_width.is_some_and(|m| iw > m)
                    || config.min_image_height.is_some_and(|m| ih < m)
                    || config.max_image_height.is_some_and(|m| ih > m))
            {
                continue;
            }
            if has_px_bounds && let Some((iw, ih)) = sz {
                // When an entry already carried a declared size and `probe_image_sizes` probed a different
                // actual size, `e.size` was the normalisation basis only for unverified COCO/VOC; otherwise
                // `sz` matches the image header directly.
                let (fw, fh) = (iw as f32, ih as f32);
                boxes.retain(|b| {
                    let bw = b.w * fw;
                    let bh = b.h * fh;
                    b.w.is_finite()
                        && b.h.is_finite()
                        && b.w >= 0.0
                        && b.h >= 0.0
                        && !out_of_bounds(bw, config.min_box_width_px, config.max_box_width_px)
                        && !out_of_bounds(bh, config.min_box_height_px, config.max_box_height_px)
                });
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn class_normalize_and_conflict_detection() {
        assert_eq!(ClassNormalize::Exact.key(" Person "), " Person ");
        assert_eq!(ClassNormalize::CaseAndWhitespace.key("  Traffic   Light \t"), "traffic light");
        assert_eq!(ClassNormalize::CaseAndWhitespace.display("  Traffic   Light "), "Traffic Light");

        let mut map = HashMap::new();
        map.insert("Car".into(), Some("vehicle".into()));
        map.insert("car".into(), Some("auto".into()));
        assert!(build_normalized_map(&map, ClassNormalize::CaseAndWhitespace).is_err());

        let mut ok_map = HashMap::new();
        ok_map.insert("Car".into(), Some("Vehicle".into()));
        ok_map.insert("car".into(), Some("vehicle".into()));
        let built = build_normalized_map(&ok_map, ClassNormalize::CaseAndWhitespace).unwrap();
        assert_eq!(built.get("car"), Some(&Some("Vehicle".into())));
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
