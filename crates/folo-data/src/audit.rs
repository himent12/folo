//! Dataset auditing: fast 128-bit content-hash duplicate detection, cross-split leakage (`train` <-> `val` / `test`)
//! reporting, and class/empty-image health checks on an opened [`LoadedDataset`].

use crate::Split;
use crate::dataset::LoadedDataset;
use crate::stats::DatasetStats;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};

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

/// An image that could not be read while auditing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnreadableImage {
    /// Split containing the entry.
    pub split: Split,
    /// 0-based index in that split.
    pub index: usize,
    /// Image path on disk.
    pub path: PathBuf,
    /// Underlying read error message.
    pub message: String,
}

/// Structured audit report produced by [`LoadedDataset::audit`] / [`audit_dataset`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AuditReport {
    /// Number of images audited per split `[train, val, test]`.
    pub splits: [usize; 3],
    /// All exact-byte duplicate groups (both intra-split and cross-split).
    pub duplicate_groups: Vec<DuplicateGroup>,
    /// Duplicate groups that span two or more distinct splits (data leakage).
    pub split_leakage: Vec<SplitLeakage>,
    /// Images that could not be read when hashing.
    pub unreadable: Vec<UnreadableImage>,
    /// Dataset label statistics across all splits.
    pub stats: DatasetStats,
    /// Class indices (into `ds.classes`) with zero boxes across the dataset.
    pub empty_classes: Vec<usize>,
}

impl AuditReport {
    /// Total redundant image copies across all duplicate groups (`sum(group.images.len() - 1)`).
    pub fn duplicate_image_count(&self) -> usize {
        self.duplicate_groups.iter().map(|g| g.images.len().saturating_sub(1)).sum()
    }

    /// True if any exact duplicate images were found (within or across splits).
    pub fn has_duplicates(&self) -> bool {
        !self.duplicate_groups.is_empty()
    }

    /// True if any exact image bytes appear in more than one split.
    pub fn has_split_leakage(&self) -> bool {
        !self.split_leakage.is_empty()
    }

    /// True if any image bytes appear in both `train` and `val`.
    pub fn has_train_val_leakage(&self) -> bool {
        self.split_leakage.iter().any(|l| l.splits.contains(&Split::Train) && l.splits.contains(&Split::Val))
    }

    /// True when no duplicates, no split leakage, and no unreadable images were found.
    pub fn is_clean(&self) -> bool {
        self.duplicate_groups.is_empty() && self.unreadable.is_empty()
    }
}

/// Audits an opened [`LoadedDataset`] for exact duplicate images, cross-split leakage, unreadable files, and
/// unused classes.
///
/// Images are hashed in parallel with a streaming 128-bit content hash (constant memory per worker thread) while
/// respecting [`crate::guard`].
pub fn audit_dataset(dataset: &LoadedDataset) -> AuditReport {
    let entries: Vec<(Split, usize, &Path)> = Split::ALL
        .into_iter()
        .flat_map(|s| {
            dataset.split(s).entries().iter().enumerate().map(move |(i, e)| (s, i, e.image.as_path()))
        })
        .collect();

    let hashed: Vec<std::result::Result<(u128, u64), String>> = entries
        .par_iter()
        .map_init(
            || vec![0u8; 64 * 1024],
            |buf, (_, _, path)| hash128_file_with_buf(path, buf).map_err(|e| e.to_string()),
        )
        .collect();

    let mut unreadable = Vec::new();
    // Preserve first-seen deterministic order of groups.
    let mut group_order: Vec<(u128, u64)> = Vec::new();
    let mut groups: HashMap<(u128, u64), Vec<AuditImageRef>> = HashMap::new();

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

    let mut duplicate_groups = Vec::new();
    let mut split_leakage = Vec::new();

    for key @ (digest, bytes) in group_order {
        let Some(images) = groups.remove(&key) else { continue };
        if images.len() < 2 {
            continue;
        }
        let hash = format!("{digest:032x}");
        let mut distinct_splits: Vec<Split> = Vec::new();
        for img in &images {
            if !distinct_splits.contains(&img.split) {
                distinct_splits.push(img.split);
            }
        }
        distinct_splits.sort();
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

    let stats = dataset.stats();
    let empty_classes = (0..dataset.classes.len())
        .filter(|&c| stats.class_counts.get(c).copied().unwrap_or(0) == 0)
        .collect();

    AuditReport {
        splits: dataset.split_sizes(),
        duplicate_groups,
        split_leakage,
        unreadable,
        stats,
        empty_classes,
    }
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
}
