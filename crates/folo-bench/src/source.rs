//! Deterministic generated-image benchmark source (docs/BENCH.md).
//!
//! Renders synthetic scenes using [`folo_data::synthetic::generate`] (FOLO's IEEE-deterministic shapes
//! generator — zero duplicate rasterizer) and encodes each frame as a canonical baseline JPEG via
//! [`crate::conditions::encode_jpeg`] with pixel `xyxy` ground-truth boxes. Same `(spec, seed)` produces
//! byte-identical images and bit-identical boxes across runs, machines, and thread counts.

use folo_api::BBox;
use folo_data::synthetic::{SyntheticConfig, generate, shapes_for};
use image::RgbImage;
use rayon::prelude::*;

use crate::conditions::encode_jpeg;

/// Default number of generated scenes when no count is given.
pub const DEFAULT_IMAGES: u32 = 64;
/// Maximum generated scenes per benchmark (matches `folo_server::bench::MAX_IMAGES`).
pub const MAX_IMAGES: u32 = 5000;
/// Default long-side image resolution in pixels.
pub const DEFAULT_IMGSZ: u32 = 640;
/// Smallest allowed image side in pixels.
pub const MIN_IMGSZ: u32 = 32;
/// Largest allowed image side in pixels.
pub const MAX_IMGSZ: u32 = 4096;
/// Largest allowed seed (`2^53 - 1`, exact in JavaScript numbers, matching `folo_server::bench::MAX_SEED`).
pub const MAX_SEED: u64 = (1u64 << 53) - 1;
/// Default class names when none are specified.
pub const DEFAULT_CLASSES: [&str; 3] = ["circle", "square", "triangle"];

/// Parameters for a deterministic generated-image benchmark source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GeneratedSourceSpec {
    /// Number of scenes to generate (`1..=5000`, default `64`).
    pub images: u32,
    /// Class names (at most `folo_data::synthetic::SHAPES.len()`, i.e. 8).
    pub classes: Vec<String>,
    /// Long side of each generated image in pixels (`32..=4096`, default `640`).
    pub imgsz: u32,
    /// Deterministic RNG seed (`0..=2^53 - 1`).
    pub seed: u64,
    /// Mix aspect ratios (`1:1`, `4:3`, `3:4`, `16:9`, `3:2`) instead of square images.
    pub vary_aspect: bool,
}

impl Default for GeneratedSourceSpec {
    fn default() -> Self {
        Self {
            images: DEFAULT_IMAGES,
            classes: DEFAULT_CLASSES.iter().map(|s| (*s).to_string()).collect(),
            imgsz: DEFAULT_IMGSZ,
            seed: 0,
            vary_aspect: false,
        }
    }
}

impl GeneratedSourceSpec {
    /// Validates every field and builds the corresponding [`SyntheticConfig`] (all images in `Split::Val` so none
    /// are wasted, and `small_fraction: 0.05` matching `RealStore::create_synthetic`).
    pub fn to_synthetic_config(&self) -> Result<SyntheticConfig, String> {
        if !(1..=MAX_IMAGES).contains(&self.images) {
            return Err(format!("images must be in 1..={MAX_IMAGES}, got {}", self.images));
        }
        if !(MIN_IMGSZ..=MAX_IMGSZ).contains(&self.imgsz) {
            return Err(format!("imgsz must be in {MIN_IMGSZ}..={MAX_IMGSZ}, got {}", self.imgsz));
        }
        if self.seed > MAX_SEED {
            return Err(format!("seed must be at most {MAX_SEED} (2^53 - 1), got {}", self.seed));
        }
        let names: Vec<String> = self.classes.iter().map(|c| c.trim().to_string()).collect();
        let shapes = shapes_for(&names).map_err(|e| e.to_string())?;
        let cfg = SyntheticConfig {
            images: self.images as usize,
            classes: shapes,
            names,
            imgsz: self.imgsz,
            seed: self.seed,
            vary_aspect: self.vary_aspect,
            val_fraction: 1.0,
            small_fraction: 0.05,
            ..SyntheticConfig::default()
        };
        cfg.check().map_err(|e| e.to_string())?;
        Ok(cfg)
    }

    /// Human-readable source label persisted as a benchmark's `source_name`.
    pub fn display_name(&self) -> String {
        let aspect = if self.vary_aspect { ", varied aspect" } else { "" };
        format!("generated ({}×{}px{aspect}, seed {})", self.images, self.imgsz, self.seed)
    }
}

/// One generated benchmark frame with its canonical JPEG bytes and pixel `xyxy` ground-truth boxes.
#[derive(Clone, Debug, PartialEq)]
pub struct GeneratedImage {
    /// Deterministic file name (`000000.jpg`, `000001.jpg`, …).
    pub file: String,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Canonical baseline JPEG bytes.
    pub jpeg: Vec<u8>,
    /// Ground-truth boxes in pixel `xyxy` coordinates, clipped to `[0, width] × [0, height]`.
    pub boxes: Vec<BBox>,
}

/// Validated handle to a deterministic generated-image source that renders bounded chunks on demand so large
/// benchmarks (`images = 5000`, `imgsz = 4096`) never buffer every encoded frame in RAM at once.
#[derive(Clone, Debug)]
pub struct GeneratedSource {
    cfg: SyntheticConfig,
}

impl GeneratedSource {
    /// Validates `spec` and returns a handle ready to render scene ranges deterministically.
    pub fn new(spec: &GeneratedSourceSpec) -> Result<Self, String> {
        Ok(Self { cfg: spec.to_synthetic_config()? })
    }

    /// Class names of the generated scenes, in class-id order.
    pub fn classes(&self) -> &[String] {
        self.cfg.class_names()
    }

    /// Total number of scenes in this source.
    pub fn len(&self) -> usize {
        self.cfg.images
    }

    /// Whether the source has zero scenes (always `false` after validation).
    pub fn is_empty(&self) -> bool {
        self.cfg.images == 0
    }

    /// Bounded parallel batch size chosen from image resolution (accounting for the `f32` raster canvas + RGB
    /// output + encoded JPEG buffers, ~16 B/px) so peak memory stays modest even at `4096×4096`.
    pub fn chunk_size(&self) -> usize {
        let side = self.cfg.imgsz as usize;
        let bytes_per_img = side.saturating_mul(side).saturating_mul(16).max(1);
        ((64 * 1024 * 1024) / bytes_per_img).clamp(1, 32)
    }

    /// Renders scenes in `range` deterministically in parallel, returning images in index order.
    pub fn render_range(&self, range: std::ops::Range<usize>) -> Result<Vec<GeneratedImage>, String> {
        if range.end > self.cfg.images {
            return Err(format!("scene range {:?} exceeds total scenes ({})", range, self.cfg.images));
        }
        range
            .into_par_iter()
            .map(|i| {
                let sample = generate(&self.cfg, i).map_err(|e| format!("generating scene {i}: {e}"))?;
                let (width, height) = (sample.image.width, sample.image.height);
                let rgb = RgbImage::from_raw(width, height, sample.image.data)
                    .ok_or_else(|| format!("generated scene {i} has unexpected pixel buffer size"))?;
                let jpeg = encode_jpeg(&rgb).map_err(|e| format!("encoding scene {i}: {e}"))?;
                let (wf, hf) = (width as f32, height as f32);
                let boxes = sample
                    .boxes
                    .into_iter()
                    .filter_map(|b| {
                        let [x1, y1, x2, y2] = b.pixel_xyxy(width, height);
                        let (x1, x2) = (x1.clamp(0.0, wf), x2.clamp(0.0, wf));
                        let (y1, y2) = (y1.clamp(0.0, hf), y2.clamp(0.0, hf));
                        (x2 - x1 >= 1.0 && y2 - y1 >= 1.0).then_some(BBox { class: b.class, x1, y1, x2, y2 })
                    })
                    .collect();
                Ok(GeneratedImage { file: format!("{i:06}.jpg"), width, height, jpeg, boxes })
            })
            .collect()
    }
}

/// Generates every scene described by `spec` deterministically in parallel using `folo-data`'s procedural shapes
/// generator. Returns `(class_names, images)` in index order (`0..spec.images`).
///
/// For large benchmarks, prefer [`GeneratedSource`] + [`GeneratedSource::render_range`] in bounded chunks so
/// encoded frames can be written incrementally instead of buffered in RAM.
pub fn generate_images(spec: &GeneratedSourceSpec) -> Result<(Vec<String>, Vec<GeneratedImage>), String> {
    let src = GeneratedSource::new(spec)?;
    let classes = src.classes().to_vec();
    let images = src.render_range(0..src.len())?;
    Ok((classes, images))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conditions::{self, Condition};

    #[test]
    fn generated_source_is_deterministic_and_feeds_conditions() {
        let spec = GeneratedSourceSpec {
            images: 6,
            classes: vec!["circle".into(), "square".into(), "triangle".into()],
            imgsz: 96,
            seed: 42,
            vary_aspect: false,
        };
        let (c1, im1) = generate_images(&spec).unwrap();
        let (c2, im2) = generate_images(&spec).unwrap();
        assert_eq!(c1, ["circle", "square", "triangle"]);
        assert_eq!(c1, c2);
        assert_eq!(im1, im2, "same spec and seed must produce byte-identical scenes");
        assert_eq!(im1.len(), 6);
        assert!(im1.iter().any(|im| !im.boxes.is_empty()), "scenes must contain ground-truth boxes");
        for (i, im) in im1.iter().enumerate() {
            assert_eq!(im.file, format!("{i:06}.jpg"));
            assert_eq!((im.width, im.height), (96, 96));
            // Every generated JPEG decodes cleanly through `folo-bench` conditions.
            let rendered = conditions::render(&im.jpeg, Condition::Haze, 2, &im.boxes, spec.seed).unwrap();
            assert_eq!((rendered.width, rendered.height), (96, 96));
            assert!(!rendered.jpeg.is_empty());
        }

        // Different seed produces different scenes.
        let other = GeneratedSourceSpec { seed: 43, ..spec.clone() };
        let (_, im3) = generate_images(&other).unwrap();
        assert_ne!(im1[0].jpeg, im3[0].jpeg);

        // Chunked streaming via `GeneratedSource::render_range` matches full generation byte for byte.
        let src = GeneratedSource::new(&spec).unwrap();
        assert_eq!(src.len(), 6);
        assert!(src.chunk_size() >= 1 && src.chunk_size() <= 32);
        let huge = GeneratedSource::new(&GeneratedSourceSpec { images: 1, imgsz: MAX_IMGSZ, ..spec.clone() })
            .unwrap();
        assert_eq!(huge.chunk_size(), 1, "4096px renders one scene per chunk to cap canvas RAM");
        let mut streamed = Vec::new();
        for start in (0..src.len()).step_by(2) {
            let end = (start + 2).min(src.len());
            streamed.extend(src.render_range(start..end).unwrap());
        }
        assert_eq!(streamed, im1);
        assert!(src.render_range(0..7).is_err());
    }

    #[test]
    fn generated_source_validates_bounds_and_classes() {
        let base = GeneratedSourceSpec::default();
        assert!(GeneratedSourceSpec { images: 0, ..base.clone() }.to_synthetic_config().is_err());
        assert!(
            GeneratedSourceSpec { images: MAX_IMAGES + 1, ..base.clone() }.to_synthetic_config().is_err()
        );
        assert!(GeneratedSourceSpec { imgsz: 16, ..base.clone() }.to_synthetic_config().is_err());
        assert!(GeneratedSourceSpec { imgsz: MAX_IMGSZ + 1, ..base.clone() }.to_synthetic_config().is_err());
        assert!(GeneratedSourceSpec { seed: MAX_SEED + 1, ..base.clone() }.to_synthetic_config().is_err());
        assert!(GeneratedSourceSpec { classes: vec![], ..base.clone() }.to_synthetic_config().is_err());
        assert!(
            GeneratedSourceSpec { classes: vec!["circle".into(), "circle".into()], ..base.clone() }
                .to_synthetic_config()
                .is_err()
        );
        // Custom non-shape class names map cleanly via `folo_data::synthetic::shapes_for`.
        let custom = GeneratedSourceSpec {
            images: 2,
            classes: vec!["bolt".into(), "nut".into()],
            imgsz: 64,
            seed: 1,
            vary_aspect: true,
        };
        let (names, imgs) = generate_images(&custom).unwrap();
        assert_eq!(names, ["bolt", "nut"]);
        assert_eq!(imgs.len(), 2);
    }
}
