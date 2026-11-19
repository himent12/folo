//! The demo's datasets: synthetic scenes drawn from a seed, with their images and labels.

use crate::api::time;
use crate::api::*;
use crate::synth::{Scene, hash};

#[derive(Clone)]
pub(in crate::api) struct DatasetFixture {
    pub(in crate::api) ds: Dataset,
    pub(in crate::api) seed: u64,
    pub(in crate::api) max_objects: u32,
    pub(in crate::api) small: bool,
    /// Per-image boxes, computed once from the scenes.
    pub(in crate::api) images: Vec<ImageInfo>,
}

impl DatasetFixture {
    pub(super) fn new(
        id: &str,
        name: &str,
        path: &str,
        format: &str,
        classes: &[&str],
        splits: (u32, u32, u32),
        max_objects: u32,
        small: bool,
        sizes: &'static [(u32, u32)],
        created: i64,
    ) -> Self {
        let seed = hash(id);
        let total = splits.0 + splits.1 + splits.2;
        let mut images = Vec::with_capacity(total as usize);
        for i in 0..total {
            let split = if i < splits.0 {
                Split::Train
            } else if i < splits.0 + splits.1 {
                Split::Val
            } else {
                Split::Test
            };
            let (w, h) = sizes[(i as usize * 7 + 3) % sizes.len()];
            let scene = Scene::generate(
                seed ^ (i as u64).wrapping_mul(0x9E37),
                classes.len() as u32,
                max_objects,
                small,
            );
            images.push(ImageInfo {
                // 1-based ids, so "7 of 4,400" in the inspector shows img_00007.
                id: format!("img_{:05}", i + 1),
                file: format!("images/{}/img_{:05}.jpg", split.label(), i + 1),
                split,
                width: w,
                height: h,
                boxes: scene.boxes(w, h),
            });
        }
        let mut class_counts = vec![0u64; classes.len()];
        let size_edges = vec![0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0];
        let mut size_counts = vec![0u64; size_edges.len() - 1];
        let per_edges = vec![0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
        let mut per_counts = vec![0u64; per_edges.len() - 1];
        let mut boxes = 0u64;
        let mut empty = 0u64;
        for im in &images {
            boxes += im.boxes.len() as u64;
            if im.boxes.is_empty() {
                empty += 1;
            }
            let n = im.boxes.len() as f64;
            let slot =
                per_edges.windows(2).position(|w| n >= w[0] && n < w[1]).unwrap_or(per_counts.len() - 1);
            per_counts[slot] += 1;
            for b in &im.boxes {
                class_counts[b.class as usize] += 1;
                // Relative size: sqrt(box area / image area), as docs/API.md defines it.
                let area = (((b.x2 - b.x1) * (b.y2 - b.y1)) as f64 / (im.width * im.height) as f64).sqrt();
                let slot = size_edges
                    .windows(2)
                    .position(|w| area >= w[0] && area < w[1])
                    .unwrap_or(size_counts.len() - 1);
                size_counts[slot] += 1;
            }
        }
        let ds = Dataset {
            id: id.into(),
            name: name.into(),
            path: path.into(),
            format: if format == "coco" { DatasetFormat::Coco } else { DatasetFormat::Yolo },
            classes: classes.iter().map(|c| c.to_string()).collect(),
            created: time::format(created),
            splits: Splits { train: splits.0 as u64, val: splits.1 as u64, test: splits.2 as u64 },
            stats: Some(DatasetStats {
                boxes,
                empty_images: empty,
                broken_labels: if small { 3 } else { 0 },
                class_counts,
                box_size_hist: Histogram { edges: size_edges, counts: size_counts },
                boxes_per_image_hist: Histogram { edges: per_edges, counts: per_counts },
            }),
            revision: None,
        };
        Self { ds, seed, max_objects, small, images }
    }

    pub(super) fn scene(&self, index: u32) -> Scene {
        Scene::generate(
            self.seed ^ (index as u64).wrapping_mul(0x9E37),
            self.ds.classes.len() as u32,
            self.max_objects,
            self.small,
        )
    }
}
