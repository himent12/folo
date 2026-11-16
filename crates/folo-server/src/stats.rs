//! Dataset statistics shared by the mock and on-disk datasets.

use folo_api::{DatasetStats, Histogram};

/// Edges of the relative box size histogram (`sqrt(box_area / image_area)`).
pub const BOX_SIZE_EDGES: [f64; 8] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
/// Edges of the boxes-per-image histogram.
pub const BOXES_PER_IMAGE_EDGES: [f64; 7] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];

/// Accumulates [`DatasetStats`] one image at a time.
pub struct StatsBuilder {
    stats: DatasetStats,
}

impl StatsBuilder {
    /// Empty stats for `classes` classes.
    pub fn new(classes: usize) -> StatsBuilder {
        StatsBuilder {
            stats: DatasetStats {
                boxes: 0,
                empty_images: 0,
                broken_labels: 0,
                class_counts: vec![0; classes],
                box_size_hist: Histogram {
                    edges: BOX_SIZE_EDGES.to_vec(),
                    counts: vec![0; BOX_SIZE_EDGES.len() - 1],
                },
                boxes_per_image_hist: Histogram {
                    edges: BOXES_PER_IMAGE_EDGES.to_vec(),
                    counts: vec![0; BOXES_PER_IMAGE_EDGES.len() - 1],
                },
            },
        }
    }

    /// Adds an image given its boxes as `(class, relative width, relative height)`.
    pub fn add_image(&mut self, boxes: impl IntoIterator<Item = (u32, f32, f32)>) {
        let s = &mut self.stats;
        let mut n = 0u64;
        for (class, w, h) in boxes {
            n += 1;
            if let Some(c) = s.class_counts.get_mut(class as usize) {
                *c += 1;
            }
            let size = (w.max(0.0) as f64 * h.max(0.0) as f64).sqrt();
            s.box_size_hist.counts[bin(&BOX_SIZE_EDGES, size)] += 1;
        }
        s.boxes += n;
        if n == 0 {
            s.empty_images += 1;
        }
        s.boxes_per_image_hist.counts[bin(&BOXES_PER_IMAGE_EDGES, n as f64)] += 1;
    }

    /// Counts labels left out because they cannot be read (`broken_labels`: one per broken label file, not per line).
    pub fn add_broken(&mut self, n: u64) {
        self.stats.broken_labels += n;
    }

    /// The result.
    pub fn finish(self) -> DatasetStats {
        self.stats
    }
}

/// Bin index of `v`: `[edges[i], edges[i+1])`, with values at or above the last edge in the last bin.
fn bin(edges: &[f64], v: f64) -> usize {
    let bins = edges.len() - 1;
    (0..bins).find(|&i| v < edges[i + 1]).unwrap_or(bins - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histograms() {
        let mut b = StatsBuilder::new(2);
        b.add_image([]);
        b.add_image([(0, 0.1, 0.1), (1, 0.5, 0.5), (7, 1.0, 1.0)]);
        b.add_broken(2);
        let s = b.finish();
        assert_eq!(s.boxes, 3);
        assert_eq!(s.empty_images, 1);
        assert_eq!(s.broken_labels, 2);
        assert_eq!(s.class_counts, vec![1, 1]);
        assert_eq!(s.box_size_hist.counts, vec![0, 0, 0, 0, 1, 0, 2]);
        assert_eq!(s.boxes_per_image_hist.counts, vec![1, 0, 1, 0, 0, 0]);
    }
}
