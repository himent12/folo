//! Dataset exact and perceptual (`dHash`) near-duplicate deduplication and pruning.
//!
//! Builds directly on [`crate::audit`]'s 128-bit content hash, 64-bit `dHash`, and exact pigeonhole multi-index
//! clustering (no duplicated hash or clustering code) and [`super::merge`]'s exclusive-create YOLO materialiser:
//! - [`deduplicate_dataset`] / [`LoadedDataset::deduplicate`]: prunes exact and/or perceptual near-duplicate
//!   groups in memory without touching files on disk, returning the pruned [`LoadedDataset`] and a structured
//!   [`DedupReport`].
//! - [`write_deduplicated`] / [`deduplicate_path`]: supports `dry_run` preview (reporting exact groups, kept item,
//!   and dropped items without creating or touching `out`) and safe materialisation into a fresh YOLO dataset
//!   directory (never mutating source images in place).

use super::CollisionPolicy;
use super::merge::{
    copy_exclusive, format_yolo_labels, resolve_merge_filenames, validate_yolo_boxes,
    write_entry_resolving_os_aliases, write_exclusive,
};
use crate::Split;
use crate::audit::{
    DEFAULT_DHASH_THRESHOLD, DecodedDhash, Dsu, ScannedContent, UnreadableImage, connect_hashes,
    exact_max_hamming, scan_dataset_images,
};
use crate::convert::mkdir;
use crate::dataset::{FileDataset, ImageEntry, LoadReport, LoadedDataset, open};
use crate::error::{DataError, Result};
use crate::image::probe_size;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn default_true() -> bool {
    true
}

fn default_dhash_threshold() -> u32 {
    DEFAULT_DHASH_THRESHOLD
}

/// Which image to keep inside each duplicate / near-duplicate group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupKeepPolicy {
    /// Keep the earliest image in deterministic split and dataset order. Default.
    #[default]
    KeepFirst,
    /// Keep the image with the largest pixel area (`width * height`); ties break by deterministic order.
    KeepLargestResolution,
    /// Keep the image with the most bounding-box annotations; ties break by deterministic order.
    KeepMostBoxes,
}

/// How duplicate / near-duplicate groups spanning multiple splits (`train`, `val`, `test`) are resolved.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CrossSplitPolicy {
    /// Prefer keeping a copy in `train` over `val` over `test` (pruning leaked copies from lower-priority
    /// splits), then choose by [`DedupKeepPolicy`] among candidates in the highest-priority split present. Default.
    #[default]
    PreferTrain,
    /// Prefer keeping a copy in `val` over `test` over `train` (protects validation/test benchmarks by pruning
    /// leaked copies from `train`).
    PreferVal,
    /// Prefer keeping a copy in `test` over `val` over `train` (protects held-out test benchmarks first).
    PreferTest,
    /// Ignore split membership and apply [`DedupKeepPolicy`] globally across all members of the group
    /// (breaking ties in `train` -> `val` -> `test`, then dataset index order).
    IgnoreSplit,
    /// Deduplicate only within each split independently; never match or drop images across different splits.
    WithinSplitOnly,
}

impl CrossSplitPolicy {
    fn split_priority_order(self) -> Option<[Split; 3]> {
        match self {
            CrossSplitPolicy::PreferTrain => Some([Split::Train, Split::Val, Split::Test]),
            CrossSplitPolicy::PreferVal => Some([Split::Val, Split::Test, Split::Train]),
            CrossSplitPolicy::PreferTest => Some([Split::Test, Split::Val, Split::Train]),
            CrossSplitPolicy::IgnoreSplit | CrossSplitPolicy::WithinSplitOnly => None,
        }
    }
}

/// How members of a [`DedupGroup`] were matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DedupMatchKind {
    /// Every member in the group has identical file bytes.
    Exact,
    /// At least two members have distinct file bytes and were linked via 64-bit `dHash` proximity.
    Perceptual,
}

/// Configuration for [`deduplicate_dataset`], [`write_deduplicated`], and [`deduplicate_path`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DedupOptions {
    /// When `true` (default), decode images and cluster perceptual near-duplicates via 64-bit `dHash` alongside
    /// exact byte duplicates. When `false`, only exact byte-identical images are grouped (no image decode).
    #[serde(default = "default_true")]
    pub perceptual: bool,
    /// Maximum Hamming distance (in `0..=64`) between two 64-bit `dHash` values to link them as perceptual
    /// near-duplicates. Values above `64` are clamped to `64`. Defaults to [`DEFAULT_DHASH_THRESHOLD`] (`6`).
    #[serde(default = "default_dhash_threshold")]
    pub max_hamming_distance: u32,
    /// Policy choosing which image to keep within each duplicate group (after applying `cross_split` priority).
    pub keep: DedupKeepPolicy,
    /// How cross-split duplicate/near-duplicate matches (`train` <-> `val` / `test`) are handled.
    pub cross_split: CrossSplitPolicy,
    /// When `true`, [`write_deduplicated`] and [`deduplicate_path`] return the full [`DedupReport`] without
    /// creating or writing any output directory or files.
    pub dry_run: bool,
    /// How filename collisions within a split are disambiguated when materialising on disk.
    pub collisions: CollisionPolicy,
}

impl Default for DedupOptions {
    fn default() -> Self {
        Self {
            perceptual: true,
            max_hamming_distance: DEFAULT_DHASH_THRESHOLD,
            keep: DedupKeepPolicy::KeepFirst,
            cross_split: CrossSplitPolicy::PreferTrain,
            dry_run: false,
            collisions: CollisionPolicy::PrefixOnCollision,
        }
    }
}

/// One image participating in a [`DedupGroup`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupItem {
    /// Split containing the image.
    pub split: Split,
    /// 0-based index within that split's [`FileDataset`].
    pub index: usize,
    /// Source image path on disk.
    pub path: PathBuf,
    /// File length in bytes.
    pub bytes: u64,
    /// Image `(width, height)` in pixels when known (decoded during perceptual scan, cached on the entry, or
    /// lazily probed for [`DedupKeepPolicy::KeepLargestResolution`]).
    pub resolution: Option<(u32, u32)>,
    /// Number of bounding-box annotations on this image.
    pub boxes: usize,
    /// 64-bit perceptual difference hash formatted as 16 lowercase hex characters (when `perceptual` is `true`
    /// and the image decoded cleanly).
    pub dhash: Option<String>,
}

/// One resolved duplicate or perceptual near-duplicate group, naming the single kept image and every dropped item.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupGroup {
    /// Whether all members share identical bytes (`Exact`) or include byte-distinct `dHash` matches (`Perceptual`).
    pub kind: DedupMatchKind,
    /// Exact maximum pairwise Hamming distance (`0..=64`) across members (`0` for exact groups).
    pub max_distance: u32,
    /// `true` when the group contained images from two or more distinct splits (cross-split leakage).
    pub cross_split: bool,
    /// Representative image retained in the deduplicated dataset.
    pub kept: DedupItem,
    /// Redundant images pruned from the dataset, in deterministic `(split, index)` order.
    pub dropped: Vec<DedupItem>,
}

/// Summary report produced by [`deduplicate_dataset`], [`write_deduplicated`], and [`deduplicate_path`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DedupReport {
    /// Whether this run was a preview-only dry run (no files written).
    pub dry_run: bool,
    /// Output YOLO root directory when materialised on disk (`None` for in-memory or `dry_run`).
    pub out_root: Option<PathBuf>,
    /// Input image counts per split `[train, val, test]`.
    pub input_splits: [usize; 3],
    /// Surviving image counts per split `[train, val, test]`.
    pub output_splits: [usize; 3],
    /// Dropped image counts per split `[train, val, test]`.
    pub dropped_splits: [usize; 3],
    /// Resolved duplicate / near-duplicate groups in deterministic first-entry order.
    pub groups: Vec<DedupGroup>,
    /// Images that could not be read (or decoded when `perceptual` is `true`).
    pub unreadable: Vec<UnreadableImage>,
    /// Number of output image filenames disambiguated on disk (`0` for in-memory or `dry_run`).
    pub renamed_collisions: usize,
}

impl DedupReport {
    /// Total input images across all splits.
    pub fn total_input(&self) -> usize {
        self.input_splits.iter().sum()
    }

    /// Total surviving images across all splits.
    pub fn total_kept(&self) -> usize {
        self.output_splits.iter().sum()
    }

    /// Total dropped images across all splits.
    pub fn total_dropped(&self) -> usize {
        self.dropped_splits.iter().sum()
    }

    /// Number of duplicate / near-duplicate groups that spanned two or more splits.
    pub fn cross_split_groups(&self) -> usize {
        self.groups.iter().filter(|g| g.cross_split).count()
    }

    /// `true` when no duplicate or near-duplicate images were found and no unreadable images occurred.
    pub fn is_clean(&self) -> bool {
        self.groups.is_empty() && self.unreadable.is_empty()
    }
}

struct ReadableNode {
    split: Split,
    index: usize,
    path: PathBuf,
    byte_key: (u128, u64),
    dhash: Option<u64>,
    resolution: Option<(u32, u32)>,
    boxes: usize,
}

/// Clusters `nodes[subset]` by exact byte hash and (when `perceptual` is `true`) by `dHash` within `threshold`,
/// returning connected components of size `>= 2` sorted deterministically by earliest node index.
fn cluster_nodes(
    nodes: &[ReadableNode],
    subset: &[usize],
    perceptual: bool,
    threshold: u32,
) -> Vec<Vec<usize>> {
    if subset.len() < 2 {
        return Vec::new();
    }
    let mut dsu = Dsu::new(subset.len());

    // 1. Union all exact byte-identical entries first.
    let mut byte_first: HashMap<(u128, u64), usize> = HashMap::with_capacity(subset.len());
    for (local_i, &node_i) in subset.iter().enumerate() {
        let key = nodes[node_i].byte_key;
        if let Some(&first) = byte_first.get(&key) {
            dsu.union(first, local_i);
        } else {
            byte_first.insert(key, local_i);
        }
    }

    // 2. When perceptual deduplication is enabled, union all entries whose dHashes are within `threshold`
    //    using `audit::connect_hashes` (exact pigeonhole multi-index hashing).
    if perceptual {
        let mut hash_to_id: HashMap<u64, usize> = HashMap::new();
        let mut unique_hashes: Vec<u64> = Vec::new();
        let mut hash_members: Vec<Vec<usize>> = Vec::new();
        for (local_i, &node_i) in subset.iter().enumerate() {
            let Some(h) = nodes[node_i].dhash else { continue };
            let hid = *hash_to_id.entry(h).or_insert_with(|| {
                let id = unique_hashes.len();
                unique_hashes.push(h);
                hash_members.push(Vec::new());
                id
            });
            hash_members[hid].push(local_i);
        }
        if unique_hashes.len() >= 2 {
            let mut hash_dsu = Dsu::new(unique_hashes.len());
            connect_hashes(&unique_hashes, threshold, &mut hash_dsu);
            let mut root_rep: HashMap<usize, usize> = HashMap::new();
            for (hid, members) in hash_members.iter().enumerate() {
                let first_member = members[0];
                for &m in &members[1..] {
                    dsu.union(first_member, m);
                }
                let hroot = hash_dsu.find(hid);
                if let Some(&prev) = root_rep.get(&hroot) {
                    dsu.union(prev, first_member);
                } else {
                    root_rep.insert(hroot, first_member);
                }
            }
        } else if let Some(members) = hash_members.first()
            && members.len() >= 2
        {
            let first = members[0];
            for &m in &members[1..] {
                dsu.union(first, m);
            }
        }
    }

    let mut root_to_comp: HashMap<usize, usize> = HashMap::new();
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for (local_i, &node_i) in subset.iter().enumerate() {
        let root = dsu.find(local_i);
        let cid = *root_to_comp.entry(root).or_insert_with(|| {
            let id = comps.len();
            comps.push(Vec::new());
            id
        });
        comps[cid].push(node_i);
    }

    let mut out: Vec<Vec<usize>> = comps
        .into_iter()
        .filter(|c| c.len() >= 2)
        .map(|mut c| {
            c.sort_unstable();
            c
        })
        .collect();
    out.sort_by_key(|c| c[0]);
    out
}

fn select_winner(
    nodes: &[ReadableNode],
    members: &[usize],
    resolutions: &HashMap<usize, Option<(u32, u32)>>,
    opts: &DedupOptions,
) -> usize {
    let candidates: Vec<usize> = match opts.cross_split.split_priority_order() {
        Some(order) => {
            let best_split = order
                .into_iter()
                .find(|&sp| members.iter().any(|&ni| nodes[ni].split == sp))
                .expect("non-empty group has at least one split");
            members.iter().copied().filter(|&ni| nodes[ni].split == best_split).collect()
        }
        None => members.to_vec(),
    };

    let mut best = candidates[0];
    match opts.keep {
        DedupKeepPolicy::KeepFirst => best,
        DedupKeepPolicy::KeepLargestResolution => {
            let area = |ni: usize| -> u64 {
                resolutions
                    .get(&ni)
                    .copied()
                    .flatten()
                    .map_or(0, |(w, h)| (w as u64).saturating_mul(h as u64))
            };
            let mut best_area = area(best);
            for &ni in &candidates[1..] {
                let a = area(ni);
                if a > best_area {
                    best = ni;
                    best_area = a;
                }
            }
            best
        }
        DedupKeepPolicy::KeepMostBoxes => {
            let mut best_boxes = nodes[best].boxes;
            for &ni in &candidates[1..] {
                let b = nodes[ni].boxes;
                if b > best_boxes {
                    best = ni;
                    best_boxes = b;
                }
            }
            best
        }
    }
}

/// Deduplicates an opened [`LoadedDataset`] in memory using [`crate::audit`]'s exact 128-bit content hash and
/// optional 64-bit `dHash` near-duplicate clustering, returning the pruned [`LoadedDataset`] and [`DedupReport`].
///
/// Source images and labels on disk are never modified. Pass the dataset to [`write_deduplicated`] to materialise
/// a clean YOLO dataset on disk (or preview with `opts.dry_run = true`).
pub fn deduplicate_dataset(
    dataset: &LoadedDataset,
    opts: &DedupOptions,
) -> Result<(LoadedDataset, DedupReport)> {
    let input_splits = dataset.split_sizes();
    let scanned = scan_dataset_images(dataset, opts.perceptual);

    let mut unreadable = Vec::new();
    let mut nodes: Vec<ReadableNode> = Vec::with_capacity(scanned.len());

    for item in scanned {
        let entry = &dataset.split(item.split).entries()[item.index];
        match item.content {
            Ok(ScannedContent { digest, bytes, perceptual }) => {
                let (dhash, resolution) = match perceptual {
                    Some(Ok(DecodedDhash { dhash, size })) => (Some(dhash), Some(size)),
                    Some(Err(message)) => {
                        unreadable.push(UnreadableImage {
                            split: item.split,
                            index: item.index,
                            path: item.path.clone(),
                            message,
                        });
                        (None, entry.size)
                    }
                    None => (None, entry.size),
                };
                nodes.push(ReadableNode {
                    split: item.split,
                    index: item.index,
                    path: item.path,
                    byte_key: (digest, bytes),
                    dhash,
                    resolution,
                    boxes: entry.boxes.len(),
                });
            }
            Err(message) => {
                unreadable.push(UnreadableImage {
                    split: item.split,
                    index: item.index,
                    path: item.path,
                    message,
                });
            }
        }
    }

    let threshold = opts.max_hamming_distance.min(64);
    let raw_components: Vec<Vec<usize>> = if opts.cross_split == CrossSplitPolicy::WithinSplitOnly {
        let mut all = Vec::new();
        for split in Split::ALL {
            let subset: Vec<usize> =
                nodes.iter().enumerate().filter(|(_, n)| n.split == split).map(|(i, _)| i).collect();
            all.extend(cluster_nodes(&nodes, &subset, opts.perceptual, threshold));
        }
        all.sort_by_key(|c| c[0]);
        all
    } else {
        let all_idxs: Vec<usize> = (0..nodes.len()).collect();
        cluster_nodes(&nodes, &all_idxs, opts.perceptual, threshold)
    };

    // Lazily resolve pixel dimensions only for members of duplicate groups when `KeepLargestResolution` is active
    // and dimensions were not already obtained during perceptual decoding.
    let mut resolutions: HashMap<usize, Option<(u32, u32)>> = HashMap::new();
    if opts.keep == DedupKeepPolicy::KeepLargestResolution {
        let to_probe: Vec<usize> =
            raw_components.iter().flatten().copied().filter(|&ni| nodes[ni].resolution.is_none()).collect();
        let probed: Vec<(usize, Option<(u32, u32)>)> = to_probe
            .par_iter()
            .map(|&ni| {
                let dims = crate::par::serial(|| probe_size(&nodes[ni].path)).ok();
                (ni, dims)
            })
            .collect();
        for &ni in raw_components.iter().flatten() {
            resolutions.insert(ni, nodes[ni].resolution);
        }
        for (ni, dims) in probed {
            if dims.is_some() {
                resolutions.insert(ni, dims);
            }
        }
    } else {
        for &ni in raw_components.iter().flatten() {
            resolutions.insert(ni, nodes[ni].resolution);
        }
    }

    let mut dropped_flags: [Vec<bool>; 3] = [
        vec![false; input_splits[Split::Train as usize]],
        vec![false; input_splits[Split::Val as usize]],
        vec![false; input_splits[Split::Test as usize]],
    ];
    let mut dropped_splits = [0usize; 3];
    let mut groups: Vec<DedupGroup> = Vec::with_capacity(raw_components.len());

    let to_item = |ni: usize, resolutions: &HashMap<usize, Option<(u32, u32)>>| -> DedupItem {
        let n = &nodes[ni];
        DedupItem {
            split: n.split,
            index: n.index,
            path: n.path.clone(),
            bytes: n.byte_key.1,
            resolution: resolutions.get(&ni).copied().flatten().or(n.resolution),
            boxes: n.boxes,
            dhash: n.dhash.map(|h| format!("{h:016x}")),
        }
    };

    for comp in raw_components {
        let first_byte_key = nodes[comp[0]].byte_key;
        let all_exact = comp.iter().all(|&ni| nodes[ni].byte_key == first_byte_key);
        let kind = if all_exact { DedupMatchKind::Exact } else { DedupMatchKind::Perceptual };
        let max_distance = if all_exact {
            0
        } else {
            let mut distinct_hashes: Vec<u64> = comp.iter().filter_map(|&ni| nodes[ni].dhash).collect();
            distinct_hashes.sort_unstable();
            distinct_hashes.dedup();
            exact_max_hamming(&distinct_hashes)
        };
        let first_split = nodes[comp[0]].split;
        let cross_split = comp.iter().any(|&ni| nodes[ni].split != first_split);

        let kept_idx = select_winner(&nodes, &comp, &resolutions, opts);
        let kept = to_item(kept_idx, &resolutions);
        let mut dropped = Vec::with_capacity(comp.len() - 1);
        for &ni in &comp {
            if ni == kept_idx {
                continue;
            }
            let n = &nodes[ni];
            dropped_flags[n.split as usize][n.index] = true;
            dropped_splits[n.split as usize] += 1;
            dropped.push(to_item(ni, &resolutions));
        }

        groups.push(DedupGroup { kind, max_distance, cross_split, kept, dropped });
    }

    let mut out_entries: [Vec<ImageEntry>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut output_splits = [0usize; 3];
    for split in Split::ALL {
        let si = split as usize;
        let src = dataset.split(split).entries();
        let flags = &dropped_flags[si];
        let mut kept = Vec::with_capacity(src.len().saturating_sub(dropped_splits[si]));
        for (i, e) in src.iter().enumerate() {
            if !flags[i] {
                kept.push(ImageEntry {
                    image: e.image.clone(),
                    label: e.label.clone(),
                    size: e.size,
                    boxes: e.boxes.clone(),
                });
            }
        }
        output_splits[si] = kept.len();
        out_entries[si] = kept;
    }

    let [train, val, test] = out_entries;
    let pruned = LoadedDataset {
        format: dataset.format,
        root: dataset.root.clone(),
        classes: dataset.classes.clone(),
        train: FileDataset::new(Split::Train, dataset.classes.clone(), train),
        val: FileDataset::new(Split::Val, dataset.classes.clone(), val),
        test: FileDataset::new(Split::Test, dataset.classes.clone(), test),
        report: LoadReport::default(),
        sources: dataset.sources.clone(),
    };
    let report = DedupReport {
        dry_run: opts.dry_run,
        out_root: None,
        input_splits,
        output_splits,
        dropped_splits,
        groups,
        unreadable,
        renamed_collisions: 0,
    };
    Ok((pruned, report))
}

/// Deduplicates `dataset` according to `opts` and, unless `opts.dry_run` is `true`, materialises a clean YOLO
/// dataset (`data.yaml` + `images/<split>/` + `labels/<split>/`) into `out` without modifying source images.
///
/// When `opts.dry_run` is `true`, `out` is neither created nor touched and the preview [`DedupReport`] is
/// returned immediately (`report.dry_run == true`, `report.out_root == None`).
///
/// When `opts.dry_run` is `false`, `out` must not exist or must be an empty directory
/// ([`crate::unfinished::claim`]). Fails with [`DataError::Config`] before creating `out` when no `train` or
/// `val` images survive, any surviving image is unreadable or has an invalid header, or any surviving entry has
/// invalid YOLO box coordinates.
pub fn write_deduplicated(dataset: &LoadedDataset, out: &Path, opts: &DedupOptions) -> Result<DedupReport> {
    let (pruned, mut report) = deduplicate_dataset(dataset, opts)?;
    if opts.dry_run {
        return Ok(report);
    }

    let [train_len, val_len, test_len] = report.output_splits;
    if train_len + val_len + test_len == 0 {
        return Err(DataError::Config(
            "nothing to write: deduplicated dataset has no surviving images".into(),
        ));
    }
    if train_len + val_len == 0 {
        return Err(DataError::Config(
            "a YOLO dataset needs train or val images, and the deduplicated dataset only has test images"
                .into(),
        ));
    }
    if let Some(bad) =
        report.unreadable.iter().find(|u| pruned.split(u.split).entries().iter().any(|e| e.image == u.path))
    {
        return Err(DataError::Config(format!(
            "cannot write deduplicated dataset with unreadable surviving image {} ({}): {}",
            bad.path.display(),
            bad.split,
            bad.message
        )));
    }
    if !opts.perceptual {
        let mut surviving: Vec<(Split, &Path)> = Vec::new();
        for split in Split::ALL {
            for e in pruned.split(split).entries() {
                surviving.push((split, e.image.as_path()));
            }
        }
        let probed: Vec<Option<(Split, &Path, DataError)>> = surviving
            .par_iter()
            .map(|&(split, path)| crate::par::serial(|| probe_size(path)).err().map(|err| (split, path, err)))
            .collect();
        if let Some((split, path, err)) = probed.into_iter().flatten().next() {
            return Err(DataError::Config(format!(
                "cannot write deduplicated dataset with unreadable surviving image {} ({}): {}",
                path.display(),
                split,
                err
            )));
        }
    }
    for split in Split::ALL {
        for e in pruned.split(split).entries() {
            validate_yolo_boxes(e, pruned.classes.len())?;
        }
    }

    let claim = crate::unfinished::claim(out)?;
    let prefixes = [String::from("img")];
    let mut renamed_collisions = 0usize;
    let mut yaml_splits = String::new();

    for split in Split::ALL {
        let entries = pruned.split(split).entries();
        if entries.is_empty() {
            continue;
        }
        let img_dir = out.join("images").join(split.as_str());
        let lbl_dir = out.join("labels").join(split.as_str());
        mkdir(&img_dir)?;
        mkdir(&lbl_dir)?;

        let items: Vec<(usize, ImageEntry)> = entries.iter().cloned().map(|e| (0usize, e)).collect();
        let (resolved, split_renames) = resolve_merge_filenames(&items, &prefixes, opts.collisions);
        renamed_collisions += split_renames;

        let has_non_ascii = resolved.iter().any(|(stem, _, _)| !stem.is_ascii());
        if has_non_ascii {
            for ((_, e), (stem, ext, already_renamed)) in items.iter().zip(&resolved) {
                if write_entry_resolving_os_aliases(e, stem, ext, &img_dir, &lbl_dir)? && !already_renamed {
                    renamed_collisions += 1;
                }
            }
        } else {
            items.par_iter().zip(&resolved).try_for_each(|((_, e), (stem, ext, _))| {
                copy_exclusive(&e.image, &img_dir.join(format!("{stem}{ext}")))
            })?;
            items.par_iter().zip(&resolved).try_for_each(|((_, e), (stem, _, _))| {
                let text = format_yolo_labels(&e.boxes);
                write_exclusive(&lbl_dir.join(format!("{stem}.txt")), text.as_bytes())
            })?;
        }

        let _ = writeln!(yaml_splits, "{split}: images/{split}");
    }

    let mut yaml = format!("# deduplicated by FOLO\n{yaml_splits}names:\n");
    for (i, c) in pruned.classes.iter().enumerate() {
        let _ = writeln!(yaml, "  {i}: {}", serde_json::to_string(c).expect("strings serialize"));
    }
    mkdir(out)?;
    write_exclusive(&out.join("data.yaml"), yaml.as_bytes())?;

    claim.finish()?;
    report.out_root = Some(out.to_path_buf());
    report.renamed_collisions = renamed_collisions;
    Ok(report)
}

/// Opens `input` (auto-detecting YOLO, COCO, or Pascal VOC), deduplicates it according to `opts`, and
/// (unless `opts.dry_run` is `true`) materialises a clean YOLO dataset at `out`.
pub fn deduplicate_path(input: impl AsRef<Path>, out: &Path, opts: &DedupOptions) -> Result<DedupReport> {
    let loaded = open(input)?;
    write_deduplicated(&loaded, out, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_options_defaults_and_serde_round_trip() {
        let def = DedupOptions::default();
        assert!(def.perceptual);
        assert_eq!(def.max_hamming_distance, DEFAULT_DHASH_THRESHOLD);
        assert_eq!(def.keep, DedupKeepPolicy::KeepFirst);
        assert_eq!(def.cross_split, CrossSplitPolicy::PreferTrain);
        assert!(!def.dry_run);
        assert_eq!(def.collisions, CollisionPolicy::PrefixOnCollision);

        let parsed: DedupOptions = serde_json::from_str(
            r#"{"keep":"keep_largest_resolution","cross_split":"prefer_val","dry_run":true}"#,
        )
        .unwrap();
        assert!(parsed.perceptual);
        assert_eq!(parsed.max_hamming_distance, DEFAULT_DHASH_THRESHOLD);
        assert_eq!(parsed.keep, DedupKeepPolicy::KeepLargestResolution);
        assert_eq!(parsed.cross_split, CrossSplitPolicy::PreferVal);
        assert!(parsed.dry_run);

        let round = serde_json::to_string(&parsed).unwrap();
        assert_eq!(serde_json::from_str::<DedupOptions>(&round).unwrap(), parsed);
    }

    #[test]
    fn select_winner_respects_split_priority_and_keep_policies() {
        let mk = |split: Split, index: usize, res: Option<(u32, u32)>, boxes: usize| ReadableNode {
            split,
            index,
            path: PathBuf::from(format!("{split}_{index}.jpg")),
            byte_key: (index as u128, 100),
            dhash: Some(0),
            resolution: res,
            boxes,
        };
        let nodes = vec![
            mk(Split::Train, 0, Some((320, 240)), 1),
            mk(Split::Train, 1, Some((640, 480)), 5),
            mk(Split::Val, 0, Some((1280, 720)), 2),
            mk(Split::Test, 0, Some((1920, 1080)), 10),
        ];
        let members = [0usize, 1, 2, 3];
        let res_map: HashMap<usize, Option<(u32, u32)>> =
            nodes.iter().enumerate().map(|(i, n)| (i, n.resolution)).collect();

        // PreferTrain filters to indices [0, 1].
        assert_eq!(
            select_winner(
                &nodes,
                &members,
                &res_map,
                &DedupOptions {
                    keep: DedupKeepPolicy::KeepFirst,
                    cross_split: CrossSplitPolicy::PreferTrain,
                    ..Default::default()
                },
            ),
            0
        );
        assert_eq!(
            select_winner(
                &nodes,
                &members,
                &res_map,
                &DedupOptions {
                    keep: DedupKeepPolicy::KeepLargestResolution,
                    cross_split: CrossSplitPolicy::PreferTrain,
                    ..Default::default()
                },
            ),
            1
        );
        // PreferVal keeps val copy (index 2).
        assert_eq!(
            select_winner(
                &nodes,
                &members,
                &res_map,
                &DedupOptions {
                    keep: DedupKeepPolicy::KeepFirst,
                    cross_split: CrossSplitPolicy::PreferVal,
                    ..Default::default()
                },
            ),
            2
        );
        // PreferTest keeps test copy (index 3).
        assert_eq!(
            select_winner(
                &nodes,
                &members,
                &res_map,
                &DedupOptions {
                    keep: DedupKeepPolicy::KeepFirst,
                    cross_split: CrossSplitPolicy::PreferTest,
                    ..Default::default()
                },
            ),
            3
        );
        // IgnoreSplit picks global highest resolution / box count (index 3).
        assert_eq!(
            select_winner(
                &nodes,
                &members,
                &res_map,
                &DedupOptions {
                    keep: DedupKeepPolicy::KeepLargestResolution,
                    cross_split: CrossSplitPolicy::IgnoreSplit,
                    ..Default::default()
                },
            ),
            3
        );
        assert_eq!(
            select_winner(
                &nodes,
                &members,
                &res_map,
                &DedupOptions {
                    keep: DedupKeepPolicy::KeepMostBoxes,
                    cross_split: CrossSplitPolicy::IgnoreSplit,
                    ..Default::default()
                },
            ),
            3
        );
    }
}
