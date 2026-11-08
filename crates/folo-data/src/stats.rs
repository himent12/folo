//! Dataset statistics, serialised exactly as the `stats` object of `docs/API.md`.

use crate::bbox::BBox;
use serde::{Deserialize, Serialize};

/// Box-size histogram edges. Size is `sqrt(w * h)` of the normalised box (a fraction of the image side).
pub const BOX_SIZE_EDGES: [f64; 8] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
/// Boxes-per-image histogram edges. The last bin also counts images with more boxes than the last edge.
pub const BOXES_PER_IMAGE_EDGES: [f64; 7] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

/// A histogram with `edges.len() - 1` bins `[edges[i], edges[i + 1])`. Values below the first edge count in the
/// first bin, values at or above the last edge in the last bin.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Histogram {
    /// Bin edges, increasing.
    pub edges: Vec<f64>,
    /// Count per bin.
    pub counts: Vec<u64>,
}

impl Histogram {
    /// An empty histogram over `edges` (at least two).
    pub fn new(edges: &[f64]) -> Self {
        assert!(edges.len() >= 2, "a histogram needs at least two edges");
        Histogram { edges: edges.to_vec(), counts: vec![0; edges.len() - 1] }
    }

    /// Counts one value.
    pub fn add(&mut self, v: f64) {
        let bins = self.counts.len();
        let i = self.edges[1..bins].partition_point(|&e| e <= v);
        self.counts[i] += 1;
    }

    /// Total count.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }
}

/// Label statistics for a dataset (`GET /api/datasets/{id}` → `stats`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total boxes.
    pub boxes: u64,
    /// Images without any box (background images).
    pub empty_images: u64,
    /// Images skipped because their labels were malformed.
    pub broken_labels: u64,
    /// Boxes per class id.
    pub class_counts: Vec<u64>,
    /// Distribution of `sqrt(w * h)` (normalised), see [`BOX_SIZE_EDGES`].
    pub box_size_hist: Histogram,
    /// Distribution of boxes per image, see [`BOXES_PER_IMAGE_EDGES`].
    pub boxes_per_image_hist: Histogram,
}

impl DatasetStats {
    /// Computes statistics from each image's boxes. Classes beyond `num_classes` grow `class_counts`.
    pub fn compute<'a>(
        num_classes: usize,
        images: impl IntoIterator<Item = &'a [BBox]>,
        broken_labels: u64,
    ) -> Self {
        let mut s = DatasetStats {
            boxes: 0,
            empty_images: 0,
            broken_labels,
            class_counts: vec![0; num_classes],
            box_size_hist: Histogram::new(&BOX_SIZE_EDGES),
            boxes_per_image_hist: Histogram::new(&BOXES_PER_IMAGE_EDGES),
        };
        for boxes in images {
            s.boxes += boxes.len() as u64;
            if boxes.is_empty() {
                s.empty_images += 1;
            }
            s.boxes_per_image_hist.add(boxes.len() as f64);
            for b in boxes {
                let c = b.class as usize;
                if c >= s.class_counts.len() {
                    s.class_counts.resize(c + 1, 0);
                }
                s.class_counts[c] += 1;
                s.box_size_hist.add((b.w.max(0.0) as f64 * b.h.max(0.0) as f64).sqrt());
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_bins() {
        let mut h = Histogram::new(&BOXES_PER_IMAGE_EDGES);
        for v in [0.0, 1.0, 1.5, 3.0, 31.0, 32.0, 500.0, -1.0] {
            h.add(v);
        }
        assert_eq!(h.counts, vec![2, 2, 1, 0, 0, 3]);
    }

    #[test]
    fn stats_and_json_shape() {
        let a = [BBox::new(0, 0.5, 0.5, 0.15, 0.15), BBox::new(2, 0.5, 0.5, 1.0, 1.0)];
        let b: [BBox; 0] = [];
        let s = DatasetStats::compute(3, [&a[..], &b[..]], 1);
        assert_eq!(s.boxes, 2);
        assert_eq!(s.empty_images, 1);
        assert_eq!(s.class_counts, vec![1, 0, 1]);
        assert_eq!(s.box_size_hist.counts, vec![0, 0, 0, 0, 1, 0, 1]);
        let json = serde_json::to_value(&s).unwrap();
        for k in [
            "boxes",
            "empty_images",
            "broken_labels",
            "class_counts",
            "box_size_hist",
            "boxes_per_image_hist",
        ] {
            assert!(json.get(k).is_some(), "{k}");
        }
        assert_eq!(json["box_size_hist"]["edges"][1], serde_json::json!(0.01));
    }
}
