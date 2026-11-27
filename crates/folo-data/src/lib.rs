//! FOLO data pipeline: datasets, image IO, augmentation and batching for training object detectors.
//!
//! folo-data does not depend on folo-core. It produces plain buffers (`Vec<f32>` images in `NCHW`, `0..1`, and
//! `[batch_idx, class, cx, cy, w, h]` targets) that the trainer wraps in tensors.
//!
//! - **Datasets** ([`open`]): YOLO (`data.yaml`, `images/` + `labels/`, segment polygons become boxes), COCO
//!   instances JSON (category ids remapped to `0..n`), and Pascal VOC XML (`Annotations/*.xml`, `JPEGImages/`,
//!   optional `ImageSets/Main/*.txt`), auto-detected. Corrupt images and broken labels are recorded in a
//!   [`LoadReport`] and skipped; JPEG EXIF orientation is applied. [`DatasetStats`] matches `docs/API.md`.
//! - **Geometry**: [`letterbox`] with exact box mapping both ways, a fixed-point, row-parallel bilinear
//!   [`resize`](resize::resize) and [`warp_affine`](warp::warp_affine).
//! - **Augmentation** ([`augment`]): mosaic, random affine, HSV, flips and mixup, all box-aware and driven by a
//!   seeded [`Rng`]; [`TrainTransform`] chains them like YOLO training (with `close_mosaic`).
//! - **Loading** ([`DataLoader`]): worker threads decode and augment ahead; batches are identical for a given seed
//!   and epoch whatever the number of workers.
//! - **Synthetic data** ([`synthetic`]): procedural shapes datasets for tests, demos and benchmarks.
//! - **Tools** ([`audit`], [`check`], [`convert`], [`curate`], [`split`], [`pack`]): exact & perceptual (`dHash`)
//!   duplicate and cross-split leakage auditing, line-level dataset validation, YOLO / COCO / Pascal VOC conversion,
//!   dataset merging, class remapping and subset filtering, deterministic multi-label stratified splitting, and a
//!   dataset as one tar stream for another FOLO server. [`unfinished`]: the marker of a folder being written.
//! - **Support**: [`guard`] (a process-wide read guard, installed by `folo mcp --confine`), [`par`] (row
//!   parallelism that the loader's workers switch off), [`paths`] (file names shared by `folo dataset check --json`
//!   and the server), [`CacheMode`] (`--cache ram`) and [`DatasetStats`].
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

pub mod audit;
pub mod augment;
mod bbox;
mod buffers;
pub mod check;
pub mod convert;
pub mod curate;
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
mod simd;
pub mod split;
mod stats;
pub mod synthetic;
pub mod unfinished;
pub mod warp;

pub use audit::{
    AuditImageRef, AuditOptions, AuditReport, DEFAULT_DHASH_THRESHOLD, DuplicateGroup, NearDuplicateGroup,
    NearDuplicateLeakage, PerceptualImageRef, SplitLeakage, UnreadableImage, audit_dataset,
    audit_dataset_with, dhash_bytes, dhash_file, dhash_image, hamming_distance,
};
pub use augment::{AugmentConfig, Fetched, TrainTransform, Transformed};
pub use bbox::BBox;
pub use convert::{
    CocoExportOptions, CocoExportReport, CocoImageMode, CocoSplitReport, VocExportOptions, VocExportReport,
    VocImageMode, VocSplitReport, export_coco, export_voc,
};
pub use curate::{
    ClassNormalize, CollisionPolicy, CrossSplitPolicy, DedupGroup, DedupItem, DedupKeepPolicy,
    DedupMatchKind, DedupOptions, DedupReport, FilterConfig, MergeConfig, MergeReport, MergeSplitPolicy,
    RemapConfig, deduplicate_dataset, deduplicate_path, filter_dataset, merge_datasets, merge_paths,
    remap_classes, write_deduplicated, write_filtered, write_merged,
};
pub use dataset::cache::{MIN_FILES as SCAN_CACHE_MIN_FILES, scan_cache_dir, set_scan_cache_dir};
pub use dataset::{
    CocoAnnotation, CocoGroundTruth, CocoImage, Dataset, FileDataset, Format, ImageEntry, InMemoryDataset,
    Issue, IssueKind, LoadReport, LoadedDataset, OpenOptions, Split, UNDETECTED, VocAnnotation, VocCoordMode,
    VocObject, VocOptions, coco_annotation_files, detect_format, label_path_for, open,
    open_coco_ground_truth, open_splits, open_voc_with, open_with, parse_voc_xml, read_coco_ground_truth,
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
pub use split::{
    SplitConfig, SplitIndices, split_dataset, split_file_dataset, split_indices, stratified_indices,
    write_split,
};
pub use stats::{BOX_SIZE_EDGES, BOXES_PER_IMAGE_EDGES, DatasetStats, Histogram};
