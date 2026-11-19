//! FOLO data pipeline: datasets, image IO, augmentation and batching for training object detectors.
//!
//! folo-data does not depend on folo-core. It produces plain buffers (`Vec<f32>` images in `NCHW`, `0..1`, and
//! `[batch_idx, class, cx, cy, w, h]` targets) that the trainer wraps in tensors.
//!
//! - **Datasets** ([`open`]): YOLO (`data.yaml`, `images/` + `labels/`, segment polygons become boxes) and COCO
//!   instances JSON (category ids remapped to `0..n`), auto-detected. Corrupt images and broken labels are recorded
//!   in a [`LoadReport`] and skipped; JPEG EXIF orientation is applied. [`DatasetStats`] matches `docs/API.md`.
//! - **Geometry**: [`letterbox`] with exact box mapping both ways, a fixed-point, row-parallel bilinear
//!   [`resize`](resize::resize) and [`warp_affine`](warp::warp_affine).
//! - **Augmentation** ([`augment`]): mosaic, random affine, HSV, flips and mixup, all box-aware and driven by a
//!   seeded [`Rng`]; [`TrainTransform`] chains them like YOLO training (with `close_mosaic`).
//! - **Loading** ([`DataLoader`]): worker threads decode and augment ahead; batches are identical for a given seed
//!   and epoch whatever the number of workers.
//! - **Synthetic data** ([`synthetic`]): procedural shapes datasets for tests, demos and benchmarks.
//! - **Tools** ([`check`], [`convert`], [`pack`]): line-level dataset validation, YOLO / COCO conversion, and a
//!   dataset as one tar stream for another FOLO server.
//! - **Support**: [`guard`] (a process-wide read guard, installed by `folo mcp --confine`), [`par`] (row
//!   parallelism that the loader's workers switch off), [`paths`] (file names shared by `folo dataset check --json`
//!   and the server), [`staging`] (a dataset folder written beside its place and renamed there when complete),
//!   [`CacheMode`] (`--cache ram`) and [`DatasetStats`].
//!
//! See `docs/DATA.md` for formats, augmentation parameters and the determinism rules.
//!
//! # Example
//!
//! ```no_run
//! use folo_data::synthetic::{SyntheticConfig, write_yolo};
//! use folo_data::{DataLoader, ImageFormat, LoaderConfig, Split, open};
//! use std::sync::Arc;
//!
//! # fn main() -> folo_data::Result<()> {
//! // Write a small shapes dataset, then read it back like any YOLO dataset.
//! let dir = std::env::temp_dir().join("folo-shapes");
//! let cfg = SyntheticConfig { images: 64, imgsz: 320, ..Default::default() };
//! write_yolo(&cfg, &dir, ImageFormat::Jpeg(90))?;
//!
//! let ds = open(&dir)?;
//! println!("{} classes, splits {:?}, {} issues", ds.classes.len(), ds.split_sizes(), ds.report.issues.len());
//! println!("{}", serde_json::to_string(&ds.stats()).unwrap());
//!
//! let loader = DataLoader::new(
//!     Arc::new(ds.split(Split::Train).clone()),
//!     LoaderConfig { batch_size: 8, imgsz: 320, seed: 7, epochs: 50, ..Default::default() },
//! )?;
//! let mut epoch = loader.epoch(0);
//! for batch in epoch.by_ref() {
//!     let batch = batch?;
//!     assert_eq!(batch.shape[1..], [3, 320, 320]);
//!     // batch.images: NCHW f32 in 0..1; batch.targets: [batch_idx, class, cx, cy, w, h]
//! }
//! println!("{:.0} images/s", epoch.throughput().images_per_s);
//! # Ok(())
//! # }
//! ```

pub mod augment;
mod bbox;
pub mod check;
pub mod convert;
pub mod dataset;
mod error;
pub mod guard;
mod image;
mod letterbox;
mod loader;
pub mod pack;
pub mod par;
pub mod paths;
mod ram_cache;
pub mod resize;
mod rng;
pub mod staging;
mod stats;
pub mod synthetic;
pub mod warp;

pub use augment::{AugmentConfig, Fetched, TrainTransform, Transformed};
pub use bbox::BBox;
pub use dataset::cache::{MIN_FILES as SCAN_CACHE_MIN_FILES, scan_cache_dir, set_scan_cache_dir};
pub use dataset::{
    CocoAnnotation, CocoGroundTruth, CocoImage, Dataset, FileDataset, Format, ImageEntry, InMemoryDataset,
    Issue, IssueKind, LoadReport, LoadedDataset, OpenOptions, Split, UNDETECTED, coco_annotation_files,
    detect_format, label_path_for, open, open_coco_ground_truth, open_splits, open_with,
    read_coco_ground_truth,
};
pub use error::{DataError, Result};
pub use image::{
    IMAGE_EXTENSIONS, Image, ImageFormat, Sample, decode, encode, is_image_path, jpeg_exif_orientation,
    jpeg_truncated, probe_size, read_image, write_image,
};
pub use letterbox::{Letterbox, LetterboxOptions, PAD_VALUE, letterbox, letterbox_sample};
pub use loader::{Batch, DataLoader, EpochIter, ImageMeta, LoaderConfig, Throughput};
pub use ram_cache::{CacheMode, RAM_CACHE_SHARE, ram_cache_bytes};
pub use rng::{Rng, mix_seed, splitmix64};
pub use stats::{BOX_SIZE_EDGES, BOXES_PER_IMAGE_EDGES, DatasetStats, Histogram};
