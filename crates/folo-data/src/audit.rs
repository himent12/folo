//! Dataset auditing: fast 128-bit content-hash duplicate detection, optional 64-bit perceptual difference-hash
//! (`dHash`) near-duplicate detection, cross-split leakage (`train` <-> `val` / `test`) reporting, and
//! class/empty-image health checks on an opened [`LoadedDataset`].

use crate::Split;
use crate::dataset::LoadedDataset;
use crate::error::Result;
use crate::image::Image;
use crate::stats::DatasetStats;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Default maximum Hamming distance (in `0..=64`) between two 64-bit `dHash` values to consider them perceptual
/// near-duplicates (~90.6% bit similarity).
pub const DEFAULT_DHASH_THRESHOLD: u32 = 6;

/// Options controlling [`audit_dataset_with`] / [`LoadedDataset::audit_with`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct AuditOptions {
    /// Decode images and compute 64-bit perceptual difference hashes (`dHash`) to catch near-duplicates
    /// (recompressed, resized, or lightly cropped images) within and across splits.
    ///
    /// Defaults to `false` in [`audit_dataset`] so the standard audit remains streaming byte-hash only.
    pub perceptual: bool,
    /// Maximum Hamming distance (in `0..=64`) between two 64-bit `dHash` values to treat them as perceptual
    /// near-duplicates. Values above `64` are clamped to `64`. Defaults to [`DEFAULT_DHASH_THRESHOLD`] (`6`).
    pub max_hamming_distance: u32,
}

impl Default for AuditOptions {
    fn default() -> Self {
        Self { perceptual: false, max_hamming_distance: DEFAULT_DHASH_THRESHOLD }
    }
}

impl AuditOptions {
    /// Enables 64-bit `dHash` perceptual near-duplicate detection with the given maximum Hamming distance.
    pub const fn perceptual(max_hamming_distance: u32) -> Self {
        Self { perceptual: true, max_hamming_distance }
    }
}

/// One image entry referenced in an [`AuditReport`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditImageRef {
    /// Split containing the image.
    pub split: Split,
    /// 0-based index in that split's [`crate::FileDataset`].
    pub index: usize,
    /// Image path on disk.
    pub path: PathBuf,
}

/// One image in a perceptual near-duplicate group, together with its 64-bit `dHash`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualImageRef {
    /// Split containing the image.
    pub split: Split,
    /// 0-based index in that split's [`crate::FileDataset`].
    pub index: usize,
    /// Image path on disk.
    pub path: PathBuf,
    /// 64-bit perceptual difference hash formatted as 16 lowercase hex characters.
    pub dhash: String,
}

/// A group of two or more images with identical file bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DuplicateGroup {
    /// 128-bit content hash formatted as 32 lowercase hex characters.
    pub hash: String,
    /// File length in bytes.
    pub bytes: u64,
    /// Matching images in deterministic split and dataset order.
    pub images: Vec<AuditImageRef>,
}

/// Content-identical images appearing in more than one split (for example `train` and `val`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitLeakage {
    /// 128-bit content hash formatted as 32 lowercase hex characters.
    pub hash: String,
    /// File length in bytes.
    pub bytes: u64,
    /// Distinct splits sharing these bytes, in split order (`train`, `val`, `test`).
    pub splits: Vec<Split>,
    /// All matching images across those splits, in deterministic split and dataset order.
    pub images: Vec<AuditImageRef>,
}

/// A single-linkage connected component of two or more perceptually similar images (recompressed, resized, or
/// lightly cropped) containing at least two distinct files by byte content.
///
/// Because grouping is transitive under [`AuditOptions::max_hamming_distance`], `max_distance` across the full
/// component can exceed the pairwise edge threshold.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearDuplicateGroup {
    /// 64-bit `dHash` of the first image in the group (16 lowercase hex characters).
    pub dhash: String,
    /// Exact maximum pairwise Hamming distance (`0..=64`) across all images in this component.
    pub max_distance: u32,
    /// Matching images in deterministic split and dataset order.
    pub images: Vec<PerceptualImageRef>,
}

/// A perceptual near-duplicate component that spans two or more distinct splits (for example a recompressed or
/// resized `train` image leaking into `val`).
///
/// Grouping is transitive (single-linkage under [`AuditOptions::max_hamming_distance`]), so `splits` lists every
/// split participating in the connected component.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NearDuplicateLeakage {
    /// 64-bit `dHash` of the first image in the group (16 lowercase hex characters).
    pub dhash: String,
    /// Exact maximum pairwise Hamming distance (`0..=64`) across all images in this component.
    pub max_distance: u32,
    /// Distinct splits sharing these near-duplicate images, in split order (`train`, `val`, `test`).
    pub splits: Vec<Split>,
    /// All matching images across those splits, in deterministic split and dataset order.
    pub images: Vec<PerceptualImageRef>,
}

/// An image that could not be read or decoded while auditing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnreadableImage {
    /// Split containing the entry.
    pub split: Split,
    /// 0-based index in that split.
    pub index: usize,
    /// Image path on disk.
    pub path: PathBuf,
    /// Underlying read or decode error message.
    pub message: String,
}

/// Structured audit report produced by [`LoadedDataset::audit`] / [`audit_dataset`] and
/// [`LoadedDataset::audit_with`] / [`audit_dataset_with`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    /// Number of images audited per split `[train, val, test]`.
    pub splits: [usize; 3],
    /// All exact-byte duplicate groups (both intra-split and cross-split).
    pub duplicate_groups: Vec<DuplicateGroup>,
    /// Exact-byte duplicate groups that span two or more distinct splits (data leakage).
    pub split_leakage: Vec<SplitLeakage>,
    /// Perceptual (`dHash`) near-duplicate groups containing at least two byte-distinct files.
    /// Empty unless [`AuditOptions::perceptual`] is enabled.
    #[serde(default)]
    pub near_duplicate_groups: Vec<NearDuplicateGroup>,
    /// Perceptual (`dHash`) near-duplicate groups that span two or more distinct splits.
    /// Empty unless [`AuditOptions::perceptual`] is enabled.
    #[serde(default)]
    pub near_duplicate_leakage: Vec<NearDuplicateLeakage>,
    /// Images that could not be read (or decoded when perceptual auditing is enabled).
    pub unreadable: Vec<UnreadableImage>,
    /// Dataset label statistics across all splits.
    pub stats: DatasetStats,
    /// Class indices (into `ds.classes`) with zero boxes across the dataset.
    pub empty_classes: Vec<usize>,
}

impl AuditReport {
    /// Total redundant image copies across all exact duplicate groups (`sum(group.images.len() - 1)`).
    pub fn duplicate_image_count(&self) -> usize {
        self.duplicate_groups.iter().map(|g| g.images.len().saturating_sub(1)).sum()
    }

    /// True if any exact duplicate images were found (within or across splits).
    pub fn has_duplicates(&self) -> bool {
        !self.duplicate_groups.is_empty()
    }

    /// True if any perceptual near-duplicate images were found (within or across splits).
    pub fn has_near_duplicates(&self) -> bool {
        !self.near_duplicate_groups.is_empty()
    }

    /// True if any perceptual near-duplicate images appear in more than one split.
    pub fn has_near_duplicate_leakage(&self) -> bool {
        !self.near_duplicate_leakage.is_empty()
    }

    /// True if any exact duplicate group or perceptual near-duplicate component spans two or more splits.
    pub fn has_split_leakage(&self) -> bool {
        !self.split_leakage.is_empty() || !self.near_duplicate_leakage.is_empty()
    }

    /// True if any exact duplicate group or perceptual near-duplicate component includes both `train` and `val`
    /// (for perceptual groups, membership is transitive across the connected component).
    pub fn has_train_val_leakage(&self) -> bool {
        self.split_leakage.iter().any(|l| l.splits.contains(&Split::Train) && l.splits.contains(&Split::Val))
            || self
                .near_duplicate_leakage
                .iter()
                .any(|l| l.splits.contains(&Split::Train) && l.splits.contains(&Split::Val))
    }

    /// True when no exact duplicates, no perceptual near-duplicates, no split leakage, and no unreadable images
    /// were found.
    pub fn is_clean(&self) -> bool {
        self.duplicate_groups.is_empty()
            && self.near_duplicate_groups.is_empty()
            && self.unreadable.is_empty()
    }
}

/// Computes the 64-bit horizontal gradient difference hash (`dHash`) of an RGB [`Image`].
///
/// Reduces the image to a `9x8` BT.601 luminance (`77*R + 150*G + 29*B`) grid by averaging integer-aligned
/// pixel tiles `[gx * w / 9, (gx + 1) * w / 9) x [gy * h / 8, (gy + 1) * h / 8)` (falling back to bilinear
/// resize when `width < 9` or `height < 8`), and sets bit `y * 8 + x` when column `x`'s mean luminance strictly
/// exceeds column `x + 1`'s (compared with exact integer cross-multiplication). Returns `0` for empty
/// (`0`-dimension) images.
///
/// # Panics
/// Panics if `image.width > 0 && image.height > 0` and `image.data` has fewer than `width * height * 3` bytes
/// (or if `width * height * 3` overflows `usize` / `isize::MAX`).
pub fn dhash_image(image: &Image) -> u64 {
    if image.width == 0 || image.height == 0 {
        return 0;
    }
    crate::simd::assert_image_data(image);
    let w = image.width as usize;
    let h = image.height as usize;
    let mut hash = 0u64;

    if w >= 9 && h >= 8 {
        let mut x_edges = [0usize; 10];
        for (gx, edge) in x_edges.iter_mut().enumerate() {
            *edge = (gx * w) / 9;
        }
        let mut y_edges = [0usize; 9];
        for (gy, edge) in y_edges.iter_mut().enumerate() {
            *edge = (gy * h) / 8;
        }
        let stride = w * 3;
        for gy in 0..8 {
            let mut sums = [0u64; 9];
            for y in y_edges[gy]..y_edges[gy + 1] {
                let row = &image.data[y * stride..(y + 1) * stride];
                for gx in 0..9 {
                    let slice = &row[x_edges[gx] * 3..x_edges[gx + 1] * 3];
                    let mut acc = 0u64;
                    for px in slice.as_chunks::<3>().0 {
                        acc += 77 * px[0] as u64 + 150 * px[1] as u64 + 29 * px[2] as u64;
                    }
                    sums[gx] += acc;
                }
            }
            for gx in 0..8 {
                let w_left = (x_edges[gx + 1] - x_edges[gx]) as u64;
                let w_right = (x_edges[gx + 2] - x_edges[gx + 1]) as u64;
                // All cells in row `gy` share the same height, so comparing `sums[gx] / w_left > sums[gx+1] / w_right`
                // is exact via cross-multiplication with no division rounding bias.
                if (sums[gx] as u128) * (w_right as u128) > (sums[gx + 1] as u128) * (w_left as u128) {
                    hash |= 1u64 << (gy * 8 + gx);
                }
            }
        }
        return hash;
    }

    let small = crate::par::serial(|| crate::resize::resize(image, 9, 8));
    let mut luma = [0u32; 9 * 8];
    for (dst, px) in luma.iter_mut().zip(small.data.as_chunks::<3>().0) {
        *dst = 77 * px[0] as u32 + 150 * px[1] as u32 + 29 * px[2] as u32;
    }
    for y in 0..8 {
        let row = &luma[y * 9..(y + 1) * 9];
        for x in 0..8 {
            if row[x] > row[x + 1] {
                hash |= 1u64 << (y * 8 + x);
            }
        }
    }
    hash
}

/// Reads and decodes the image at `path` (honouring EXIF orientation and [`crate::guard`]) and computes its
/// 64-bit perceptual difference hash ([`dhash_image`]).
pub fn dhash_file(path: &Path) -> Result<u64> {
    let img = crate::read_image(path)?;
    Ok(dhash_image(&img))
}

/// Hamming distance (number of differing bits, `0..=64`) between two 64-bit perceptual hashes.
#[inline]
pub const fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Audits an opened [`LoadedDataset`] for exact duplicate images, cross-split leakage, unreadable files, and
/// unused classes.
///
/// Images are hashed in parallel with a streaming 128-bit content hash (constant memory per worker thread) while
/// respecting [`crate::guard`]. Use [`audit_dataset_with`] with [`AuditOptions::perceptual`] to also detect
/// perceptual (`dHash`) near-duplicates across recompressions, resizes, and light crops.
pub fn audit_dataset(dataset: &LoadedDataset) -> AuditReport {
    audit_dataset_with(dataset, &AuditOptions::default())
}

/// Audits an opened [`LoadedDataset`] with configurable options (exact byte hashing plus optional 64-bit
/// `dHash` perceptual near-duplicate and cross-split leakage detection).
pub fn audit_dataset_with(dataset: &LoadedDataset, opts: &AuditOptions) -> AuditReport {
    let entries: Vec<(Split, usize, &Path)> = Split::ALL
        .into_iter()
        .flat_map(|s| {
            dataset.split(s).entries().iter().enumerate().map(move |(i, e)| (s, i, e.image.as_path()))
        })
        .collect();

    let mut unreadable = Vec::new();
    let mut group_order: Vec<(u128, u64)> = Vec::new();
    let mut groups: HashMap<(u128, u64), Vec<AuditImageRef>> = HashMap::new();
    let mut perceptual_entries: Vec<PerceptualEntry> = Vec::new();

    if opts.perceptual {
        struct ScannedFile {
            digest: u128,
            len: u64,
            dhash: std::result::Result<u64, String>,
        }
        let scanned: Vec<std::result::Result<ScannedFile, String>> = entries
            .par_iter()
            .map(|(_, _, path)| {
                crate::guard::check(path).map_err(|e| e.to_string())?;
                let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
                let digest = hash128(&bytes);
                let len = bytes.len() as u64;
                let dhash = crate::par::serial(|| {
                    crate::image::decode_with_path(&bytes, path)
                        .map(|img| dhash_image(&img))
                        .map_err(|e| e.to_string())
                });
                Ok(ScannedFile { digest, len, dhash })
            })
            .collect();

        for ((split, index, path), res) in entries.into_iter().zip(scanned) {
            match res {
                Ok(ScannedFile { digest, len, dhash }) => {
                    let key = (digest, len);
                    groups
                        .entry(key)
                        .or_insert_with(|| {
                            group_order.push(key);
                            Vec::new()
                        })
                        .push(AuditImageRef { split, index, path: path.to_path_buf() });
                    match dhash {
                        Ok(dhash) => {
                            perceptual_entries.push(PerceptualEntry {
                                split,
                                index,
                                path: path.to_path_buf(),
                                byte_key: key,
                                dhash,
                            });
                        }
                        Err(message) => {
                            unreadable.push(UnreadableImage {
                                split,
                                index,
                                path: path.to_path_buf(),
                                message,
                            });
                        }
                    }
                }
                Err(message) => {
                    unreadable.push(UnreadableImage { split, index, path: path.to_path_buf(), message });
                }
            }
        }
    } else {
        let hashed: Vec<std::result::Result<(u128, u64), String>> = entries
            .par_iter()
            .map_init(
                || vec![0u8; 64 * 1024],
                |buf, (_, _, path)| hash128_file_with_buf(path, buf).map_err(|e| e.to_string()),
            )
            .collect();

        for ((split, index, path), res) in entries.into_iter().zip(hashed) {
            match res {
                Ok(key) => {
                    groups
                        .entry(key)
                        .or_insert_with(|| {
                            group_order.push(key);
                            Vec::new()
                        })
                        .push(AuditImageRef { split, index, path: path.to_path_buf() });
                }
                Err(message) => {
                    unreadable.push(UnreadableImage { split, index, path: path.to_path_buf(), message });
                }
            }
        }
    }

    let mut duplicate_groups = Vec::new();
    let mut split_leakage = Vec::new();

    for key @ (digest, bytes) in group_order {
        let Some(images) = groups.remove(&key) else { continue };
        if images.len() < 2 {
            continue;
        }
        let hash = format!("{digest:032x}");
        let distinct_splits = distinct_splits_of(images.iter().map(|i| i.split));
        if distinct_splits.len() >= 2 {
            split_leakage.push(SplitLeakage {
                hash: hash.clone(),
                bytes,
                splits: distinct_splits,
                images: images.clone(),
            });
        }
        duplicate_groups.push(DuplicateGroup { hash, bytes, images });
    }

    let (near_duplicate_groups, near_duplicate_leakage) = if opts.perceptual {
        cluster_perceptual(&perceptual_entries, opts.max_hamming_distance.min(64))
    } else {
        (Vec::new(), Vec::new())
    };

    let stats = dataset.stats();
    let empty_classes = (0..dataset.classes.len())
        .filter(|&c| stats.class_counts.get(c).copied().unwrap_or(0) == 0)
        .collect();

    AuditReport {
        splits: dataset.split_sizes(),
        duplicate_groups,
        split_leakage,
        near_duplicate_groups,
        near_duplicate_leakage,
        unreadable,
        stats,
        empty_classes,
    }
}

struct PerceptualEntry {
    split: Split,
    index: usize,
    path: PathBuf,
    byte_key: (u128, u64),
    dhash: u64,
}

fn distinct_splits_of(splits: impl Iterator<Item = Split>) -> Vec<Split> {
    let mut distinct: Vec<Split> = Vec::new();
    for s in splits {
        if !distinct.contains(&s) {
            distinct.push(s);
        }
    }
    distinct.sort();
    distinct
}

/// Disjoint-set union over `0..n`.
struct Dsu {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self { parent: (0..n).collect(), rank: vec![0; n] }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra == rb {
            return;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
    }
}

/// Connects unique hashes within `threshold` Hamming distance using O(m) fast path at `threshold >= 64`, exact
/// pigeonhole multi-index hashing for large sets with small thresholds (`threshold <= 7`), or direct pairwise
/// scan otherwise.
fn connect_hashes(hashes: &[u64], threshold: u32, dsu: &mut Dsu) {
    let m = hashes.len();
    if m < 2 || threshold == 0 {
        return;
    }
    if threshold >= 64 {
        for u in 1..m {
            dsu.union(0, u);
        }
        return;
    }
    if m <= 2048 || threshold > 7 {
        for u in 0..m {
            let hu = hashes[u];
            for (v, &hv) in hashes.iter().enumerate().skip(u + 1) {
                if hamming_distance(hu, hv) <= threshold {
                    dsu.union(u, v);
                }
            }
        }
        return;
    }
    // Pigeonhole principle: partitioning 64 bits into `threshold + 1` disjoint bands guarantees any two hashes
    // with Hamming distance <= `threshold` match identically in at least one band.
    let num_bands = (threshold + 1) as usize;
    let mut buckets: HashMap<u64, Vec<usize>> = HashMap::new();
    for b in 0..num_bands {
        buckets.clear();
        let bit_start = (b * 64) / num_bands;
        let bit_end = ((b + 1) * 64) / num_bands;
        let bit_len = bit_end - bit_start;
        let mask = (1u64 << bit_len) - 1;
        for (idx, &h) in hashes.iter().enumerate() {
            let key = (h >> bit_start) & mask;
            buckets.entry(key).or_default().push(idx);
        }
        for idxs in buckets.values() {
            if idxs.len() < 2 {
                continue;
            }
            for i in 0..idxs.len() {
                let u = idxs[i];
                let hu = hashes[u];
                for &v in &idxs[i + 1..] {
                    if hamming_distance(hu, hashes[v]) <= threshold {
                        dsu.union(u, v);
                    }
                }
            }
        }
    }
}

/// Computes the exact maximum pairwise Hamming distance across a slice of distinct 64-bit hashes, stopping early
/// once the theoretical upper bound `64` is reached.
fn exact_max_hamming(hashes: &[u64]) -> u32 {
    if hashes.len() < 2 {
        return 0;
    }
    if hashes.len() <= 1024 {
        let mut best = 0u32;
        for (i, &ha) in hashes.iter().enumerate() {
            for &hb in &hashes[i + 1..] {
                let d = hamming_distance(ha, hb);
                if d > best {
                    best = d;
                    if best == 64 {
                        return 64;
                    }
                }
            }
        }
        return best;
    }
    (0..hashes.len())
        .into_par_iter()
        .map(|i| {
            let ha = hashes[i];
            let mut local = 0u32;
            for &hb in &hashes[i + 1..] {
                let d = hamming_distance(ha, hb);
                if d > local {
                    local = d;
                    if local == 64 {
                        break;
                    }
                }
            }
            local
        })
        .reduce(|| 0u32, u32::max)
}

/// Clusters `entries` by `dHash` within `threshold` Hamming distance, excluding groups whose members are all
/// exact byte-identical copies (which are already reported in `duplicate_groups` / `split_leakage`).
fn cluster_perceptual(
    entries: &[PerceptualEntry],
    threshold: u32,
) -> (Vec<NearDuplicateGroup>, Vec<NearDuplicateLeakage>) {
    if entries.len() < 2 {
        return (Vec::new(), Vec::new());
    }

    // Deduplicate by exact dHash first so identical perceptual hashes share a single node.
    let mut hash_to_node: HashMap<u64, usize> = HashMap::new();
    let mut unique_hashes: Vec<u64> = Vec::new();
    let mut node_entries: Vec<Vec<usize>> = Vec::new();
    for (entry_idx, e) in entries.iter().enumerate() {
        let node = *hash_to_node.entry(e.dhash).or_insert_with(|| {
            let id = unique_hashes.len();
            unique_hashes.push(e.dhash);
            node_entries.push(Vec::new());
            id
        });
        node_entries[node].push(entry_idx);
    }

    let mut dsu = Dsu::new(unique_hashes.len());
    connect_hashes(&unique_hashes, threshold, &mut dsu);

    // Gather connected components in deterministic first-entry order.
    let mut root_to_comp: HashMap<usize, usize> = HashMap::new();
    let mut comp_entries: Vec<Vec<usize>> = Vec::new();
    let mut comp_hashes: Vec<Vec<u64>> = Vec::new();
    for (node, members) in node_entries.into_iter().enumerate() {
        let root = dsu.find(node);
        let comp = *root_to_comp.entry(root).or_insert_with(|| {
            let id = comp_entries.len();
            comp_entries.push(Vec::new());
            comp_hashes.push(Vec::new());
            id
        });
        comp_entries[comp].extend(members);
        comp_hashes[comp].push(unique_hashes[node]);
    }

    // Sort components deterministically by their earliest entry index.
    let mut ordered: Vec<(Vec<usize>, Vec<u64>)> = comp_entries
        .into_iter()
        .zip(comp_hashes)
        .filter(|(m, _)| m.len() >= 2)
        .map(|(mut m, h)| {
            m.sort_unstable();
            (m, h)
        })
        .collect();
    ordered.sort_by_key(|(m, _)| m[0]);

    let mut near_groups = Vec::new();
    let mut near_leakage = Vec::new();

    for (member_idxs, hashes) in ordered {
        // Skip clusters where every image has the exact same file bytes (already in exact duplicate_groups).
        let first_byte_key = entries[member_idxs[0]].byte_key;
        if member_idxs.iter().all(|&i| entries[i].byte_key == first_byte_key) {
            continue;
        }

        // Exact maximum pairwise Hamming distance across distinct hashes in the component.
        let max_distance = exact_max_hamming(&hashes);

        let rep_dhash = format!("{:016x}", entries[member_idxs[0]].dhash);
        let images: Vec<PerceptualImageRef> = member_idxs
            .iter()
            .map(|&i| {
                let e = &entries[i];
                PerceptualImageRef {
                    split: e.split,
                    index: e.index,
                    path: e.path.clone(),
                    dhash: format!("{:016x}", e.dhash),
                }
            })
            .collect();

        let distinct_splits = distinct_splits_of(images.iter().map(|i| i.split));
        if distinct_splits.len() >= 2 {
            near_leakage.push(NearDuplicateLeakage {
                dhash: rep_dhash.clone(),
                max_distance,
                splits: distinct_splits,
                images: images.clone(),
            });
        }

        near_groups.push(NearDuplicateGroup { dhash: rep_dhash, max_distance, images });
    }

    (near_groups, near_leakage)
}

/// Streaming 128-bit content hasher (two independent 64-bit multiply-rotate lanes over 8-byte words).
pub(crate) struct Hasher128 {
    a: u64,
    b: u64,
    tail: [u8; 8],
    tail_len: usize,
    total_len: u64,
}

impl Hasher128 {
    const K1: u64 = 0x9e37_79b9_7f4a_7c15;
    const K2: u64 = 0xc2b2_ae3d_27d4_eb4f;

    pub(crate) fn new() -> Self {
        Self { a: 0x243f_6a88_85a3_08d3, b: 0x1319_8a2e_0370_7344, tail: [0; 8], tail_len: 0, total_len: 0 }
    }

    #[inline]
    fn mix_word(&mut self, w: [u8; 8]) {
        let v = u64::from_le_bytes(w);
        self.a = (self.a ^ v).wrapping_mul(Self::K1).rotate_left(29);
        self.b = (self.b ^ v.rotate_left(32)).wrapping_mul(Self::K2).rotate_left(31);
    }

    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.total_len = self.total_len.wrapping_add(bytes.len() as u64);
        if self.tail_len > 0 {
            let need = 8 - self.tail_len;
            if bytes.len() < need {
                self.tail[self.tail_len..self.tail_len + bytes.len()].copy_from_slice(bytes);
                self.tail_len += bytes.len();
                return;
            }
            self.tail[self.tail_len..8].copy_from_slice(&bytes[..need]);
            let w = self.tail;
            self.tail_len = 0;
            self.mix_word(w);
            bytes = &bytes[need..];
        }
        let (words, rem) = bytes.as_chunks::<8>();
        for &w in words {
            self.mix_word(w);
        }
        if !rem.is_empty() {
            self.tail[..rem.len()].copy_from_slice(rem);
            self.tail_len = rem.len();
        }
    }

    pub(crate) fn finish(self) -> u128 {
        let mut last = [0u8; 8];
        last[..self.tail_len].copy_from_slice(&self.tail[..self.tail_len]);
        let v = u64::from_le_bytes(last) ^ ((self.tail_len as u64) << 56);
        let a = crate::splitmix64(&mut ((self.a ^ self.total_len) ^ v));
        let b =
            crate::splitmix64(&mut ((self.b ^ self.total_len.rotate_left(32)) ^ v.wrapping_mul(Self::K1)));
        ((a as u128) << 64) | b as u128
    }
}

/// Fast 128-bit content hash over an in-memory byte slice.
pub(crate) fn hash128(bytes: &[u8]) -> u128 {
    let mut h = Hasher128::new();
    h.update(bytes);
    h.finish()
}

/// Streams a file through [`Hasher128`] using a reusable scratch buffer, returning `(hash, byte_len)`.
fn hash128_file_with_buf(path: &Path, buf: &mut [u8]) -> std::io::Result<(u128, u64)> {
    crate::guard::check(path)?;
    let mut file = std::fs::File::open(path)?;
    let mut h = Hasher128::new();
    let mut total = 0u64;
    loop {
        match file.read(buf) {
            Ok(0) => break,
            Ok(n) => {
                h.update(&buf[..n]);
                total += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok((h.finish(), total))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_hash_matches_one_shot_across_chunk_sizes() {
        let data: Vec<u8> = (0..1027).map(|i| (i * 37 + 11) as u8).collect();
        let expected = hash128(&data);
        for chunk in [1, 3, 7, 8, 9, 13, 64, 511, 1027] {
            let mut h = Hasher128::new();
            for part in data.chunks(chunk) {
                h.update(part);
            }
            assert_eq!(h.finish(), expected, "chunk size {chunk}");
        }
        assert_ne!(hash128(b"abc"), hash128(b"abd"));
        assert_ne!(hash128(b""), hash128(b"\0"));
        assert_ne!(hash128(&[0u8; 8]), hash128(&[0u8; 9]));
    }

    #[test]
    fn dhash_invariance_and_multi_index_matches_bruteforce() {
        // Empty image returns 0 without panicking.
        assert_eq!(dhash_image(&Image { width: 0, height: 0, data: Vec::new() }), 0);

        // Structured gradient + block image vs downscaled version have tiny Hamming distance;
        // horizontally inverted image has large Hamming distance.
        let mut img = Image::new(64, 48, [0, 0, 0]);
        for y in 0..48 {
            for x in 0..64 {
                let r = ((x * 4) as u8).saturating_add(if y > 24 { 40 } else { 0 });
                let g = ((x * 3 + y * 2) % 255) as u8;
                let b = (255u32.saturating_sub(x * 3)) as u8;
                img.set_pixel(x, y, [r, g, b]);
            }
        }
        let h_orig = dhash_image(&img);
        let h_resized = dhash_image(&crate::resize::resize(&img, 32, 24));
        assert!(
            hamming_distance(h_orig, h_resized) <= 2,
            "resized distance {} too large ({h_orig:016x} vs {h_resized:016x})",
            hamming_distance(h_orig, h_resized)
        );

        let mut inv = Image::new(64, 48, [0, 0, 0]);
        for y in 0..48 {
            for x in 0..64 {
                let p = img.pixel(63 - x, y);
                inv.set_pixel(x, y, p);
            }
        }
        let h_inv = dhash_image(&inv);
        assert!(hamming_distance(h_orig, h_inv) >= 20, "dist={}", hamming_distance(h_orig, h_inv));

        // Verify pigeonhole multi-index hashing (> 2048 unique hashes) produces identical connected components
        // to brute-force O(M^2) for thresholds 1..=7.
        let mut rng = crate::Rng::new(2026);
        let mut hashes: Vec<u64> = (0..2100).map(|_| rng.next_u64()).collect();
        // Plant near-duplicate pairs at distances 1, 3, 6, 7 across indices.
        for &(src, dst, flips) in &[(10, 1500, 1u32), (20, 2099, 3), (30, 1024, 6), (40, 2050, 7)] {
            let mut h = hashes[src];
            for b in 0..flips {
                h ^= 1u64 << (b * 9 % 64);
            }
            hashes[dst] = h;
        }
        for threshold in [1u32, 3, 6, 7] {
            let mut dsu_mih = Dsu::new(hashes.len());
            connect_hashes(&hashes, threshold, &mut dsu_mih);

            let mut dsu_bf = Dsu::new(hashes.len());
            for u in 0..hashes.len() {
                for v in (u + 1)..hashes.len() {
                    if hamming_distance(hashes[u], hashes[v]) <= threshold {
                        dsu_bf.union(u, v);
                    }
                }
            }
            // Canonicalize component membership by earliest index per component.
            let canon = |dsu: &mut Dsu| -> Vec<usize> {
                let mut root_min: HashMap<usize, usize> = HashMap::new();
                for i in 0..hashes.len() {
                    let r = dsu.find(i);
                    root_min.entry(r).or_insert(i);
                }
                (0..hashes.len()).map(|i| root_min[&dsu.find(i)]).collect()
            };
            assert_eq!(canon(&mut dsu_mih), canon(&mut dsu_bf), "mismatch at threshold {threshold}");
        }

        // Small image (< 9x8) exercises the bilinear fallback path cleanly.
        let tiny = Image::new(4, 3, [120, 80, 40]);
        assert_eq!(dhash_image(&tiny), 0);
        let mut tiny_grad = Image::new(5, 4, [0, 0, 0]);
        for y in 0..4 {
            for x in 0..5 {
                let v = (255 - x * 50) as u8;
                tiny_grad.set_pixel(x, y, [v, v, v]);
            }
        }
        assert_eq!(dhash_image(&tiny_grad), u64::MAX);

        // Exact diameter beyond 512 unique hashes (including complementary pair at index 512).
        let mut big: Vec<u64> = (0..512u64).collect();
        big.push(!511u64);
        assert_eq!(exact_max_hamming(&big), 64);

        // AuditOptions deserializes with #[serde(default)] when max_hamming_distance is omitted.
        let opts: AuditOptions = serde_json::from_str(r#"{"perceptual": true}"#).unwrap();
        assert_eq!(opts, AuditOptions::perceptual(DEFAULT_DHASH_THRESHOLD));
    }

    #[test]
    #[should_panic(expected = "image buffer")]
    fn dhash_image_panics_on_truncated_buffer() {
        let bad = Image { width: 9, height: 8, data: Vec::new() };
        let _ = dhash_image(&bad);
    }
}
