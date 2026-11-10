//! Where a benchmark's images (with their truth) and a loop's pool come from: a split of a registered dataset or
//! a labeling job, read through the [`Store`], so a benchmark is made of exactly what the dataset or the job
//! shows.

use std::path::PathBuf;

use axum::http::StatusCode;
use folo_api::{BBox, BenchSource, Split};

use crate::error::{Error, Result};
use crate::store::{ImageQuery, Store};
use crate::util::Rng;

/// An image of a source.
#[derive(Clone, Debug)]
pub struct SourceImage {
    /// Its file name (the last part of its path).
    pub file: String,
    /// Where the file is.
    pub path: PathBuf,
    /// Width after EXIF orientation.
    pub width: u32,
    /// Height.
    pub height: u32,
    /// Its boxes (a dataset's labels, a labeled job image's boxes; empty for a pending job image).
    pub boxes: Vec<BBox>,
}

/// What a source holds.
pub struct Gathered {
    /// The dataset's or the job's name.
    pub name: String,
    /// Its classes.
    pub classes: Vec<String>,
    /// Its images, in the source's order (a dataset split in path order, a job in the order images were added).
    pub images: Vec<SourceImage>,
    /// The split read (datasets).
    pub split: Option<Split>,
}

/// The images of `source`: a dataset's `split` (`default_split` when it names none), or a labeling job's images
/// (with `truth`, only the labeled ones). With `truth` the images carry their boxes; without, they carry none
/// (a loop's pool: [`pool`]). `404 dataset_not_found` / `label_job_not_found` for a source that does not exist.
pub fn gather(
    store: &dyn Store,
    source: &BenchSource,
    default_split: Split,
    truth: bool,
) -> Result<Gathered> {
    match source {
        BenchSource::Dataset { dataset_id, split } => {
            let split = split.unwrap_or(default_split);
            let ds = store.get_dataset(dataset_id)?;
            let q = ImageQuery { split: Some(split), offset: 0, limit: u64::MAX, class: None };
            let page = store.list_images(dataset_id, &q)?;
            let images = page
                .images
                .into_iter()
                .map(|im| {
                    let path = store.dataset_image_path(dataset_id, &im.id)?;
                    let file = im.file.rsplit(['/', '\\']).next().unwrap_or(&im.file).to_string();
                    let boxes = if truth { im.boxes } else { Vec::new() };
                    Ok(SourceImage { file, path, width: im.width, height: im.height, boxes })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Gathered { name: ds.name, classes: ds.classes, images, split: Some(split) })
        }
        BenchSource::LabelJob { job_id } => {
            let job = store.labeler()?.job_images(job_id, truth)?;
            let images = job
                .images
                .into_iter()
                .map(|im| SourceImage {
                    file: im.file,
                    path: im.path,
                    width: im.width,
                    height: im.height,
                    boxes: if truth { im.boxes } else { Vec::new() },
                })
                .collect();
            Ok(Gathered { name: job.name, classes: job.classes, images, split: None })
        }
    }
}

/// An image of a self-improvement loop's pool: a file and its size, never its labels (a pool's own labels are
/// never read, so they are not even carried).
#[derive(Clone, Debug)]
pub struct PoolImage {
    /// Its file name.
    pub file: String,
    /// Where the file is.
    pub path: PathBuf,
    /// Width after EXIF orientation.
    pub width: u32,
    /// Height.
    pub height: u32,
}

/// A loop's pool.
pub struct Pool {
    /// The dataset's or the job's name.
    pub name: String,
    /// Its images, in the source's order.
    pub images: Vec<PoolImage>,
    /// The split read (datasets; default `train`).
    pub split: Option<Split>,
}

/// The images of a loop's pool: a dataset's split (`train` by default) or every image of a labeling job, without
/// their labels.
pub fn pool(store: &dyn Store, source: &BenchSource) -> Result<Pool> {
    let g = gather(store, source, Split::Train, false)?;
    let images = g
        .images
        .into_iter()
        .map(|im| PoolImage { file: im.file, path: im.path, width: im.width, height: im.height })
        .collect();
    Ok(Pool { name: g.name, images, split: g.split })
}

/// `400 invalid_dataset`: a source with nothing to use.
pub fn empty(message: impl Into<String>, hint: &str) -> Error {
    Error::new(StatusCode::BAD_REQUEST, "invalid_dataset", message).hint(hint)
}

/// Which of `n` source images a benchmark of at most `max` takes, in its order: a seeded shuffle of all of them,
/// cut to `max`. So the sample (when the source has more) and the order are one draw, and any prefix of the
/// order is itself a fair sample.
pub fn sample_order(n: usize, max: usize, seed: u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed ^ 0xB3_4C_5A_3B);
    for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i as u64 + 1) as usize);
    }
    order.truncate(max);
    order
}

/// True boxes as a benchmark keeps them: clipped to the `width` x `height` image, corners in order; a box with a
/// coordinate that is not a number, a class outside `0..classes`, or less than a pixel of width or height left is
/// dropped. So a benchmark's box count is exactly what is scored.
pub fn clip_truth(boxes: &[BBox], classes: usize, (width, height): (u32, u32)) -> Vec<BBox> {
    let (w, h) = (width as f32, height as f32);
    boxes
        .iter()
        .filter(|b| (b.class as usize) < classes && [b.x1, b.y1, b.x2, b.y2].iter().all(|v| v.is_finite()))
        .filter_map(|b| {
            let (x1, x2) = (b.x1.min(b.x2).clamp(0.0, w), b.x1.max(b.x2).clamp(0.0, w));
            let (y1, y2) = (b.y1.min(b.y2).clamp(0.0, h), b.y1.max(b.y2).clamp(0.0, h));
            (x2 - x1 >= 1.0 && y2 - y1 >= 1.0).then_some(BBox { class: b.class, x1, y1, x2, y2 })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_are_seeded_shuffles_and_prefixes() {
        let a = sample_order(100, 100, 7);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..100).collect::<Vec<_>>());
        assert_ne!(a, (0..100).collect::<Vec<_>>(), "shuffled");
        assert_eq!(sample_order(100, 100, 7), a, "seeded");
        assert_ne!(sample_order(100, 100, 8), a);
        // A smaller benchmark of the same source and seed is the first images of the larger one.
        assert_eq!(sample_order(100, 10, 7), a[..10]);
        assert_eq!(sample_order(3, 10, 7).len(), 3);
        assert!(sample_order(0, 10, 7).is_empty());
    }

    #[test]
    fn truth_is_clipped_and_degenerate_boxes_dropped() {
        let b = |class, x1, y1, x2, y2| BBox { class, x1, y1, x2, y2 };
        let kept = clip_truth(
            &[
                b(0, -5.0, 2.0, 50.0, 30.0),
                b(1, 90.0, 40.0, 60.0, 10.0),
                b(2, 0.0, 0.0, 10.0, 10.0),
                b(0, f32::NAN, 0.0, 10.0, 10.0),
                b(0, 99.5, 0.0, 120.0, 10.0),
                b(1, 10.0, 10.0, 10.5, 20.0),
            ],
            2,
            (100, 50),
        );
        assert_eq!(kept, [b(0, 0.0, 2.0, 50.0, 30.0), b(1, 60.0, 10.0, 90.0, 40.0)]);
    }
}
