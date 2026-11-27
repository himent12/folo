//! Deterministic multi-label stratified dataset splitting (`train` / `val` / optional `test`).
//!
//! Uses iterative multi-label stratification (Sechidis, Tsoumakas & Vlahavas, 2011) with relative-deficit scoring so
//! rare classes are distributed across active splits first, multi-label images keep balanced class proportions, and
//! background (box-less) images are stratified proportionally too. Every tie-break is driven by [`crate::Rng`], so
//! splits are bit-for-bit reproducible for a given [`SplitConfig::seed`].

use crate::convert::{Converted, convert};
use crate::dataset::{Dataset, FileDataset, ImageEntry, LoadReport, LoadedDataset};
use crate::error::{DataError, Result};
use crate::rng::Rng;
use crate::{Format, Split};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Configuration for deterministic multi-label stratified splitting.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SplitConfig {
    /// Fraction of images assigned to `train` (finite, in `[0.0, 1.0]`).
    pub train: f64,
    /// Fraction of images assigned to `val` (finite, in `[0.0, 1.0]`).
    pub val: f64,
    /// Fraction of images assigned to `test` (finite, in `[0.0, 1.0]`).
    pub test: f64,
    /// Seed driving sample ordering and tie-breaking.
    pub seed: u64,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self::train_val(0.8, 0.2, 0)
    }
}

impl SplitConfig {
    /// Two-way `train` / `val` split (`test = 0.0`).
    pub const fn train_val(train: f64, val: f64, seed: u64) -> Self {
        Self { train, val, test: 0.0, seed }
    }

    /// Three-way `train` / `val` / `test` split.
    pub const fn train_val_test(train: f64, val: f64, test: f64, seed: u64) -> Self {
        Self { train, val, test, seed }
    }

    /// Ratios as `[train, val, test]`.
    pub const fn ratios(&self) -> [f64; 3] {
        [self.train, self.val, self.test]
    }

    /// Validates that ratios are finite, non-negative, and sum to `1.0` within `1e-6`.
    pub fn validate(&self) -> Result<()> {
        let ratios = self.ratios();
        for (name, r) in [("train", ratios[0]), ("val", ratios[1]), ("test", ratios[2])] {
            if !r.is_finite() || !(0.0..=1.0).contains(&r) {
                return Err(DataError::Config(format!(
                    "split ratio `{name}` must be finite and in [0.0, 1.0], got {r}"
                )));
            }
        }
        let sum: f64 = ratios.iter().sum();
        if !sum.is_finite() || (sum - 1.0).abs() > 1e-6 {
            return Err(DataError::Config(format!(
                "split ratios train ({}) + val ({}) + test ({}) must sum to 1.0, got {sum:.6}",
                self.train, self.val, self.test
            )));
        }
        Ok(())
    }
}

/// 0-based sample indices assigned to each split by [`stratified_indices`] or [`split_indices`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitIndices {
    /// Sample indices assigned to `train` (sorted ascending).
    pub train: Vec<usize>,
    /// Sample indices assigned to `val` (sorted ascending).
    pub val: Vec<usize>,
    /// Sample indices assigned to `test` (sorted ascending).
    pub test: Vec<usize>,
}

impl SplitIndices {
    /// Counts `[train, val, test]`.
    pub fn counts(&self) -> [usize; 3] {
        [self.train.len(), self.val.len(), self.test.len()]
    }

    /// Indices for one split.
    pub fn for_split(&self, split: Split) -> &[usize] {
        match split {
            Split::Train => &self.train,
            Split::Val => &self.val,
            Split::Test => &self.test,
        }
    }
}

/// Computes deterministic multi-label stratified split indices for any [`Dataset`].
pub fn split_indices<D: Dataset + ?Sized>(dataset: &D, config: &SplitConfig) -> Result<SplitIndices> {
    let labels: Vec<Vec<u32>> =
        (0..dataset.len()).map(|i| dataset.boxes(i).iter().map(|b| b.class).collect()).collect();
    stratified_indices(&labels, config)
}

/// Computes deterministic multi-label stratified split indices from per-sample class lists.
///
/// Each element of `sample_classes` lists the class IDs present in that sample (duplicates are ignored; an empty
/// slice is treated as a background sample and stratified proportionally as its own stratum).
pub fn stratified_indices(sample_classes: &[Vec<u32>], config: &SplitConfig) -> Result<SplitIndices> {
    config.validate()?;
    let n = sample_classes.len();
    if n == 0 {
        return Err(DataError::Config("cannot split an empty dataset (0 samples)".into()));
    }

    let raw = config.ratios();
    let sum: f64 = raw.iter().sum();
    let ratios = [raw[0] / sum, raw[1] / sum, raw[2] / sum];
    let active: Vec<usize> = (0..3).filter(|&s| ratios[s] > 0.0).collect();
    let mut cap = target_capacities(n, &ratios, &active);

    // Build deduplicated label sets per sample; empty samples get a dedicated virtual background label.
    let mut class_to_idx: BTreeMap<Option<u32>, usize> = BTreeMap::new();
    let mut sample_labels: Vec<Vec<usize>> = Vec::with_capacity(n);
    for classes in sample_classes {
        let mut uniq: Vec<usize> = Vec::new();
        if classes.is_empty() {
            let next = class_to_idx.len();
            let id = *class_to_idx.entry(None).or_insert(next);
            uniq.push(id);
        } else {
            for &c in classes {
                let next = class_to_idx.len();
                let id = *class_to_idx.entry(Some(c)).or_insert(next);
                if !uniq.contains(&id) {
                    uniq.push(id);
                }
            }
        }
        sample_labels.push(uniq);
    }
    let num_labels = class_to_idx.len();

    // Seeded permutation of sample indices so every tie and iteration order is deterministic and unbiased.
    let mut rng = Rng::new(config.seed);
    let mut order: Vec<usize> = (0..n).collect();
    rng.shuffle(&mut order);

    // Per-label list of remaining sample indices (in shuffled order) and desired counts per split.
    let mut label_samples: Vec<Vec<usize>> = vec![Vec::new(); num_labels];
    for &idx in &order {
        for &l in &sample_labels[idx] {
            label_samples[l].push(idx);
        }
    }
    let mut desired: Vec<[f64; 3]> = label_samples
        .iter()
        .map(|s| {
            let total = s.len() as f64;
            [total * ratios[0], total * ratios[1], total * ratios[2]]
        })
        .collect();
    let mut remaining_per_label: Vec<usize> = label_samples.iter().map(Vec::len).collect();
    let mut assigned = vec![false; n];
    let mut buckets: [Vec<usize>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut remaining_total = n;

    while remaining_total > 0 {
        // Pick the rarest label with at least one unassigned sample (smallest count wins).
        let Some(rarest) = (0..num_labels)
            .filter(|&l| remaining_per_label[l] > 0)
            .min_by_key(|&l| (remaining_per_label[l], l))
        else {
            break;
        };

        // Iterate through the unassigned samples carrying `rarest`.
        let samples = std::mem::take(&mut label_samples[rarest]);
        for idx in samples {
            if assigned[idx] {
                continue;
            }
            let chosen = pick_split(rarest, &sample_labels[idx], &desired, &ratios, &cap, &active, &mut rng);
            assigned[idx] = true;
            remaining_total -= 1;
            cap[chosen] -= 1;
            buckets[chosen].push(idx);
            for &l in &sample_labels[idx] {
                remaining_per_label[l] -= 1;
                desired[l][chosen] -= 1.0;
            }
        }
    }

    for b in &mut buckets {
        b.sort_unstable();
    }
    let [train, val, test] = buckets;
    Ok(SplitIndices { train, val, test })
}

/// Computes integer split capacities summing to `n` using largest remainder, guaranteeing every active split gets at
/// least 1 sample whenever `n >= active.len()`.
fn target_capacities(n: usize, ratios: &[f64; 3], active: &[usize]) -> [usize; 3] {
    let mut cap = [0usize; 3];
    let mut frac = [(0usize, 0.0f64); 3];
    let mut used = 0usize;
    for &s in active {
        let exact = n as f64 * ratios[s];
        let floor = (exact.floor() as usize).min(n);
        cap[s] = floor;
        used += floor;
        frac[s] = (s, exact - floor as f64);
    }
    let mut order: Vec<usize> = active.to_vec();
    order.sort_by(|&a, &b| {
        frac[b].1.total_cmp(&frac[a].1).then_with(|| ratios[b].total_cmp(&ratios[a])).then_with(|| a.cmp(&b))
    });
    while used > n {
        if let Some(&donor) = active.iter().filter(|&&d| cap[d] > 0).max_by_key(|&&d| cap[d]) {
            cap[donor] -= 1;
            used -= 1;
        } else {
            break;
        }
    }
    let mut rem = n.saturating_sub(used);
    let mut cursor = 0usize;
    while rem > 0 {
        let s = order[cursor % order.len()];
        cap[s] += 1;
        rem -= 1;
        cursor += 1;
    }

    // If there are enough samples for every active split to receive at least one, ensure none is starved to 0.
    if n >= active.len() {
        for &s in active {
            if cap[s] == 0
                && let Some(&donor) = active.iter().filter(|&&d| cap[d] > 1).max_by_key(|&&d| cap[d])
            {
                cap[donor] -= 1;
                cap[s] += 1;
            }
        }
    }
    debug_assert_eq!(cap.iter().sum::<usize>(), n);
    cap
}

/// Chooses the best split with remaining capacity for a sample carrying `label` and `sample_labels`.
///
/// Maximises relative deficit `desired[label][s] / ratio[s]`, which gives every active split a representative of a
/// rare class (with >= 2 samples) before another split takes a second, while converging to exact target ratios for
/// larger counts. Ties break by the sample's combined relative deficit across all its other labels (preserving
/// co-occurring class balance when singletons appear), then by largest remaining split capacity, then deterministically
/// via `rng`.
fn pick_split(
    label: usize,
    sample_labels: &[usize],
    desired: &[[f64; 3]],
    ratios: &[f64; 3],
    cap: &[usize; 3],
    active: &[usize],
    rng: &mut Rng,
) -> usize {
    const EPS: f64 = 1e-9;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_splits = [0usize; 3];
    let mut best_len = 0usize;

    for &s in active {
        if cap[s] == 0 {
            continue;
        }
        let score = desired[label][s] / ratios[s];
        if best_len == 0 || score > best_score + EPS {
            best_score = score;
            best_splits[0] = s;
            best_len = 1;
        } else if (score - best_score).abs() <= EPS {
            best_splits[best_len] = s;
            best_len += 1;
        }
    }
    debug_assert!(best_len > 0);
    if best_len == 1 {
        return best_splits[0];
    }

    // Tie-breaker 1: combined relative deficit across the sample's other labels (handles multi-label co-occurrence).
    let mut co_best = f64::NEG_INFINITY;
    let mut co_splits = [0usize; 3];
    let mut co_len = 0usize;
    for &s in &best_splits[..best_len] {
        let co_score: f64 =
            sample_labels.iter().filter(|&&l| l != label).map(|&l| desired[l][s] / ratios[s]).sum();
        if co_len == 0 || co_score > co_best + EPS {
            co_best = co_score;
            co_splits[0] = s;
            co_len = 1;
        } else if (co_score - co_best).abs() <= EPS {
            co_splits[co_len] = s;
            co_len += 1;
        }
    }
    if co_len == 1 {
        return co_splits[0];
    }

    // Tie-breaker 2: largest remaining capacity (sends true singletons to train rather than val/test).
    let max_cap = co_splits[..co_len].iter().map(|&s| cap[s]).max().unwrap_or(0);
    let mut tied = [0usize; 3];
    let mut tied_len = 0usize;
    for &s in &co_splits[..co_len] {
        if cap[s] == max_cap {
            tied[tied_len] = s;
            tied_len += 1;
        }
    }
    if tied_len == 1 { tied[0] } else { tied[rng.below(tied_len)] }
}

/// Re-splits a [`LoadedDataset`] (pooling all entries across its existing `train`, `val`, and `test` splits)
/// into deterministic multi-label stratified `train`, `val`, and `test` splits.
pub fn split_dataset(dataset: &LoadedDataset, config: &SplitConfig) -> Result<LoadedDataset> {
    let entries: Vec<ImageEntry> =
        Split::ALL.into_iter().flat_map(|s| dataset.split(s).entries().iter().cloned()).collect();
    let labels: Vec<Vec<u32>> = entries.iter().map(|e| e.boxes.iter().map(|b| b.class).collect()).collect();
    let indices = stratified_indices(&labels, config)?;

    let take = |split: Split, idxs: &[usize]| {
        let split_entries: Vec<ImageEntry> = idxs.iter().map(|&i| entries[i].clone()).collect();
        FileDataset::new(split, dataset.classes.clone(), split_entries)
    };
    Ok(LoadedDataset {
        format: dataset.format,
        root: dataset.root.clone(),
        classes: dataset.classes.clone(),
        train: take(Split::Train, &indices.train),
        val: take(Split::Val, &indices.val),
        test: take(Split::Test, &indices.test),
        report: LoadReport::default(),
        sources: dataset.sources.clone(),
    })
}

/// Splits a single [`FileDataset`] into `(train, val, test)` [`FileDataset`]s using multi-label stratification.
pub fn split_file_dataset(
    dataset: &FileDataset,
    config: &SplitConfig,
) -> Result<(FileDataset, FileDataset, FileDataset)> {
    let indices = split_indices(dataset, config)?;
    let classes = dataset.class_names().to_vec();
    let entries = dataset.entries();
    let take = |split: Split, idxs: &[usize]| {
        FileDataset::new(split, classes.clone(), idxs.iter().map(|&i| entries[i].clone()).collect())
    };
    Ok((take(Split::Train, &indices.train), take(Split::Val, &indices.val), take(Split::Test, &indices.test)))
}

/// Splits `dataset` via [`split_dataset`] and materialises the result on disk in format `to` (`Yolo`, `Coco`, or
/// `Voc`) using [`convert`].
pub fn write_split(
    dataset: &LoadedDataset,
    out: &Path,
    to: Format,
    config: &SplitConfig,
) -> Result<Converted> {
    let split = split_dataset(dataset, config)?;
    convert(&split, out, to)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_ratios_and_empty_inputs() {
        assert!(SplitConfig::train_val(0.8, 0.2, 0).validate().is_ok());
        assert!(SplitConfig::train_val_test(0.7, 0.2, 0.1, 0).validate().is_ok());
        assert!(SplitConfig::train_val(0.8, 0.3, 0).validate().is_err());
        assert!(SplitConfig::train_val(-0.1, 1.1, 0).validate().is_err());
        assert!(SplitConfig::train_val(f64::NAN, 1.0, 0).validate().is_err());
        assert!(SplitConfig::train_val(f64::INFINITY, 0.0, 0).validate().is_err());
        assert!(stratified_indices(&[], &SplitConfig::default()).is_err());
    }

    #[test]
    fn preserves_rare_classes_and_exact_ratios() {
        // 100 images: 90 with class 0, 8 with class 1 (multi-label with 0), 2 with rare class 2, 10 background.
        let mut labels = Vec::new();
        for _ in 0..80 {
            labels.push(vec![0]);
        }
        for _ in 0..8 {
            labels.push(vec![0, 1]);
        }
        for _ in 0..2 {
            labels.push(vec![0, 2]);
        }
        for _ in 0..10 {
            labels.push(vec![]);
        }
        let cfg = SplitConfig::train_val(0.8, 0.2, 42);
        let split = stratified_indices(&labels, &cfg).unwrap();
        assert_eq!(split.counts(), [80, 20, 0]);
        // Same seed is bit-for-bit deterministic.
        assert_eq!(split, stratified_indices(&labels, &cfg).unwrap());
        // Different seed produces a different partition.
        assert_ne!(split, stratified_indices(&labels, &SplitConfig::train_val(0.8, 0.2, 99)).unwrap());
        // Rare class 2 (2 samples) must appear in BOTH train and val.
        let count_class = |idxs: &[usize], c: u32| idxs.iter().filter(|&&i| labels[i].contains(&c)).count();
        assert_eq!(count_class(&split.train, 2), 1);
        assert_eq!(count_class(&split.val, 2), 1);
        // Class 1 (8 samples): ~6 train, ~2 val.
        assert!((5..=7).contains(&count_class(&split.train, 1)));
        assert!((1..=3).contains(&count_class(&split.val, 1)));
        // Background images (10 samples): 8 train, 2 val.
        let bg = |idxs: &[usize]| idxs.iter().filter(|&&i| labels[i].is_empty()).count();
        assert_eq!(bg(&split.train), 8);
        assert_eq!(bg(&split.val), 2);
    }

    #[test]
    fn three_way_split_and_small_dataset_edge_cases() {
        // 30 samples, 3 classes of 10 each, 70/20/10 split.
        let labels: Vec<Vec<u32>> = (0..30).map(|i| vec![(i % 3) as u32]).collect();
        let split = stratified_indices(&labels, &SplitConfig::train_val_test(0.7, 0.2, 0.1, 7)).unwrap();
        assert_eq!(split.counts(), [21, 6, 3]);
        for c in 0..3u32 {
            assert_eq!(split.test.iter().filter(|&&i| labels[i] == [c]).count(), 1);
            assert_eq!(split.val.iter().filter(|&&i| labels[i] == [c]).count(), 2);
            assert_eq!(split.train.iter().filter(|&&i| labels[i] == [c]).count(), 7);
        }
        // Single sample goes to largest split (train).
        let one = stratified_indices(&[vec![0]], &SplitConfig::train_val(0.8, 0.2, 1)).unwrap();
        assert_eq!(one.counts(), [1, 0, 0]);
        // Two samples with 80/20 still give val 1 image instead of starving it.
        let two = stratified_indices(&[vec![0], vec![0]], &SplitConfig::train_val(0.8, 0.2, 1)).unwrap();
        assert_eq!(two.counts(), [1, 1, 0]);
    }

    #[test]
    fn handles_tolerance_boundary_ratios_and_cooccurring_singletons() {
        // Ratios within 1e-6 of 1.0 must normalize cleanly and never panic or underflow at large n.
        for (cfg, n) in [
            (SplitConfig::train_val(0.999_999_1, 0.0, 0), 2_500_000usize),
            (SplitConfig::train_val(0.8, 0.199_999_1, 0), 2_222_223usize),
            (SplitConfig::train_val(0.500_000_4, 0.500_000_4, 0), 2_500_000usize),
        ] {
            cfg.validate().unwrap();
            let raw = cfg.ratios();
            let sum: f64 = raw.iter().sum();
            let norm = [raw[0] / sum, raw[1] / sum, raw[2] / sum];
            let active: Vec<usize> = (0..3).filter(|&s| norm[s] > 0.0).collect();
            let cap = target_capacities(n, &norm, &active);
            assert_eq!(cap.iter().sum::<usize>(), n, "cfg={cfg:?}");
        }

        // Co-occurrence: 8 backgrounds + {A, B} + {A, C} at 80/20 must spread A across both train and val.
        let mut labels = vec![vec![0, 1], vec![0, 2]];
        labels.extend((0..8).map(|_| vec![]));
        let split = stratified_indices(&labels, &SplitConfig::train_val(0.8, 0.2, 11)).unwrap();
        assert_eq!(split.counts(), [8, 2, 0]);
        let a_train = split.train.iter().filter(|&&i| labels[i].contains(&0)).count();
        let a_val = split.val.iter().filter(|&&i| labels[i].contains(&0)).count();
        assert_eq!((a_train, a_val), (1, 1), "class A must appear in both train and val");
    }
}
