//! Procedural shapes datasets: deterministic, no downloads, good-looking.
//!
//! Each image is a textured or gradient background with a handful of anti-aliased shapes (circle, square,
//! triangle, star, ring, cross, hexagon, heart) in random colours, sizes (including small objects), rotations and
//! positions, with soft shadows, partial occlusion, objects cut by the frame and some empty images. Everything is a
//! pure function of `(config, seed, index)`.
//!
//! Use [`write_yolo`] to write a standard YOLO folder (`images/`, `labels/`, `data.yaml`) or [`SyntheticDataset`]
//! to render samples on demand in memory.
//!
//! `raster.rs` holds the anti-aliased signed-distance rasteriser and the procedural backgrounds, `math.rs` the sine,
//! cosine and powers of two they use: built from IEEE basic operations, so a seed gives the same bytes in every
//! build and on every machine (the generator takes no approximation from the platform's libm; `sqrt` is
//! correctly rounded, `floor`, `ceil`, `round` and a remainder are exact).

mod math;
mod raster;

use crate::bbox::BBox;
use crate::dataset::{Dataset, Split};
use crate::error::{DataError, Result};
use crate::image::{ImageFormat, Sample, write_image};
use crate::rng::Rng;
use raster::{Background, Canvas, Paint, Prim, bounds_of, hsv_to_rgb, luma};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::consts::{PI, TAU};
use std::path::{Path, PathBuf};

/// Shape classes the generator can draw.
pub const SHAPES: [&str; 8] = ["circle", "square", "triangle", "star", "ring", "cross", "hexagon", "heart"];

/// Generator settings. The JSON shape accepts the `POST /api/datasets/synthetic` body (`images`, `classes`,
/// `imgsz`, `seed`); the other fields have defaults.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SyntheticConfig {
    /// Number of images (train + val).
    pub images: usize,
    /// Class names, each one of [`SHAPES`]; class ids follow this order.
    pub classes: Vec<String>,
    /// Long side of each image in pixels.
    pub imgsz: u32,
    /// Seed; the same seed gives byte-identical datasets.
    pub seed: u64,
    /// Fewest objects in a non-empty image.
    pub min_objects: usize,
    /// Most objects in an image.
    pub max_objects: usize,
    /// Fraction of images with no objects.
    pub empty_fraction: f32,
    /// Fraction of objects that are small (8–28 px across).
    pub small_fraction: f32,
    /// Largest fraction of an earlier object's box a later object may cover (partial occlusion).
    pub max_occlusion: f32,
    /// Fraction of objects allowed to be partly outside the frame.
    pub edge_fraction: f32,
    /// Use a mix of aspect ratios (1:1, 4:3, 3:4, 16:9, 3:2) instead of square images.
    pub vary_aspect: bool,
    /// Fraction of images in the validation split.
    pub val_fraction: f32,
    /// Sensor grain amplitude in 8-bit levels.
    pub grain: f32,
    /// Display names written to `data.yaml` and returned by [`Dataset::class_names`], one per entry of `classes`
    /// (empty: the shape names themselves). [`shapes_for`] picks shapes for arbitrary names.
    pub names: Vec<String>,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        SyntheticConfig {
            images: 200,
            classes: SHAPES[..6].iter().map(|s| s.to_string()).collect(),
            imgsz: 640,
            seed: 0,
            min_objects: 1,
            max_objects: 8,
            empty_fraction: 0.05,
            small_fraction: 0.2,
            max_occlusion: 0.35,
            edge_fraction: 0.1,
            vary_aspect: true,
            val_fraction: 0.2,
            grain: 5.0,
            names: Vec::new(),
        }
    }
}

impl SyntheticConfig {
    /// Fails with [`DataError::Config`] when the generator cannot use this config.
    pub fn check(&self) -> Result<()> {
        self.kinds().map(drop)
    }

    /// Checks the settings and resolves class names to shape kinds.
    fn kinds(&self) -> Result<Vec<usize>> {
        if self.classes.is_empty() {
            return Err(DataError::Config("synthetic dataset needs at least one class".into()));
        }
        if self.imgsz < 32 {
            return Err(DataError::Config(format!(
                "synthetic imgsz must be at least 32, got {}",
                self.imgsz
            )));
        }
        if self.min_objects > self.max_objects {
            return Err(DataError::Config("min_objects must not exceed max_objects".into()));
        }
        let fractions = [
            ("empty_fraction", self.empty_fraction),
            ("small_fraction", self.small_fraction),
            ("max_occlusion", self.max_occlusion),
            ("edge_fraction", self.edge_fraction),
            ("val_fraction", self.val_fraction),
        ];
        for (name, f) in fractions {
            if !(0.0..=1.0).contains(&f) {
                return Err(DataError::Config(format!("{name} must be in [0, 1], got {f}")));
            }
        }
        let mut kinds = Vec::with_capacity(self.classes.len());
        for c in &self.classes {
            let k = SHAPES.iter().position(|s| s.eq_ignore_ascii_case(c)).ok_or_else(|| {
                DataError::Config(format!("unknown synthetic class {c:?}; available: {}", SHAPES.join(", ")))
            })?;
            if kinds.contains(&k) {
                return Err(DataError::Config(format!("class {c:?} is listed twice")));
            }
            kinds.push(k);
        }
        if !self.names.is_empty() {
            if self.names.len() != self.classes.len() {
                return Err(DataError::Config(format!(
                    "{} names for {} classes",
                    self.names.len(),
                    self.classes.len()
                )));
            }
            check_names(&self.names)?;
        }
        Ok(kinds)
    }

    /// The class names of the dataset: `names`, or the shape names when it is empty.
    pub fn class_names(&self) -> &[String] {
        if self.names.is_empty() { &self.classes } else { &self.names }
    }

    /// Number of training images; the rest are validation.
    pub fn train_count(&self) -> usize {
        let val = (self.images as f64 * self.val_fraction as f64).round() as usize;
        self.images - val.min(self.images)
    }

    /// The global image indices of a split.
    pub fn split_range(&self, split: Split) -> std::ops::Range<usize> {
        match split {
            Split::Train => 0..self.train_count(),
            Split::Val => self.train_count()..self.images,
            Split::Test => self.images..self.images,
        }
    }
}

fn check_names(names: &[impl AsRef<str>]) -> Result<()> {
    for (i, n) in names.iter().enumerate() {
        let n = n.as_ref();
        if n.trim().is_empty() {
            return Err(DataError::Config("class names must not be empty".into()));
        }
        if names[..i].iter().any(|m| m.as_ref() == n) {
            return Err(DataError::Config(format!("class {n:?} is listed twice")));
        }
    }
    Ok(())
}

/// Shapes for arbitrary class names, for [`SyntheticConfig::classes`] (with the names in
/// [`SyntheticConfig::names`]): a name that is a shape (any case) gets that shape, every other name the first shape
/// in [`SHAPES`] order that no name has claimed. At most `SHAPES.len()` distinct, non-empty names.
pub fn shapes_for(names: &[impl AsRef<str>]) -> Result<Vec<String>> {
    if names.is_empty() {
        return Err(DataError::Config("synthetic dataset needs at least one class".into()));
    }
    if names.len() > SHAPES.len() {
        return Err(DataError::Config(format!(
            "a synthetic dataset has at most {} classes (one per shape), got {}",
            SHAPES.len(),
            names.len()
        )));
    }
    check_names(names)?;
    let shape_of = |n: &str| SHAPES.iter().position(|s| s.eq_ignore_ascii_case(n.trim()));
    let mut taken = [false; SHAPES.len()];
    for k in names.iter().filter_map(|n| shape_of(n.as_ref())) {
        if taken[k] {
            return Err(DataError::Config(format!("two classes name the shape {:?}", SHAPES[k])));
        }
        taken[k] = true;
    }
    Ok(names
        .iter()
        .map(|n| {
            let k = shape_of(n.as_ref()).unwrap_or_else(|| {
                let k = taken.iter().position(|t| !t).expect("at most SHAPES.len() names");
                taken[k] = true;
                k
            });
            SHAPES[k].to_string()
        })
        .collect())
}

struct Object {
    class: u32,
    prims: Vec<Prim>,
    hue: f32,
    sat: f32,
    val: f32,
}

/// A laid-out image: everything needed to render it and its labels.
struct Scene {
    width: u32,
    height: u32,
    background: Background,
    objects: Vec<Object>,
    boxes: Vec<BBox>,
    grain_seed: u64,
}

fn rotate_pts(pts: &[[f32; 2]], c: [f32; 2], r: f32, rot: f32) -> Vec<[f32; 2]> {
    let (s, co) = math::sin_cos(rot);
    pts.iter().map(|p| [c[0] + r * (p[0] * co - p[1] * s), c[1] + r * (p[0] * s + p[1] * co)]).collect()
}

fn regular(n: usize, phase: f32) -> Vec<[f32; 2]> {
    (0..n)
        .map(|i| {
            let (s, c) = math::sin_cos(phase + TAU * i as f32 / n as f32);
            [c, s]
        })
        .collect()
}

/// Primitives of a shape of kind `k` with circumradius `r` at `c`, rotated by `rot`.
fn shape_prims(k: usize, c: [f32; 2], r: f32, rot: f32, rng: &mut Rng) -> Vec<Prim> {
    match SHAPES[k] {
        "circle" => vec![Prim::Disc { c, r }],
        "ring" => {
            let t = rng.uniform(0.16, 0.3);
            vec![Prim::Annulus { c, mid: r * (1.0 - t), half: r * t }]
        }
        "square" => vec![Prim::Poly(rotate_pts(&regular(4, PI / 4.0), c, r, rot))],
        "triangle" => vec![Prim::Poly(rotate_pts(&regular(3, -PI / 2.0), c, r, rot))],
        "hexagon" => vec![Prim::Poly(rotate_pts(&regular(6, 0.0), c, r, rot))],
        "star" => {
            let inner = rng.uniform(0.38, 0.5);
            let pts: Vec<[f32; 2]> = (0..10)
                .map(|i| {
                    let (s, c) = math::sin_cos(-PI / 2.0 + PI * i as f32 / 5.0);
                    let rr = if i % 2 == 0 { 1.0 } else { inner };
                    [rr * c, rr * s]
                })
                .collect();
            vec![Prim::Poly(rotate_pts(&pts, c, r, rot))]
        }
        "cross" => {
            let a = rng.uniform(0.24, 0.34);
            let pts = [
                [-a, -1.0],
                [a, -1.0],
                [a, -a],
                [1.0, -a],
                [1.0, a],
                [a, a],
                [a, 1.0],
                [-a, 1.0],
                [-a, a],
                [-1.0, a],
                [-1.0, -a],
                [-a, -a],
            ];
            vec![Prim::Poly(rotate_pts(&pts, c, r * std::f32::consts::FRAC_1_SQRT_2 * 1.3, rot))]
        }
        _ => {
            // Heart, point down: a diamond plus two discs (an exact union), centred on its bounding box.
            let s = r / 0.6;
            let diamond = [[0.0, 0.552], [0.5, 0.052], [0.0, -0.448], [-0.5, 0.052]];
            let mut prims = vec![Prim::Poly(rotate_pts(&diamond, c, s, rot))];
            for lc in rotate_pts(&[[-0.25, -0.198], [0.25, -0.198]], c, s, rot) {
                prims.push(Prim::Disc { c: lc, r: std::f32::consts::SQRT_2 / 4.0 * s });
            }
            prims
        }
    }
}

fn layout(cfg: &SyntheticConfig, kinds: &[usize], index: usize) -> Scene {
    let mut rng = Rng::derive(&[cfg.seed, index as u64, 0x5CE4E]);
    let long = cfg.imgsz as f32;
    let (aw, ah) = if cfg.vary_aspect {
        [(1.0, 1.0), (4.0, 3.0), (3.0, 4.0), (16.0, 9.0), (3.0, 2.0), (1.0, 1.0)][rng.below(6)]
    } else {
        (1.0, 1.0)
    };
    let scale = long / f32::max(aw, ah);
    let (width, height) = (((aw * scale).round() as u32).max(16), ((ah * scale).round() as u32).max(16));
    let (w, h) = (width as f32, height as f32);

    // Background: muted colours so the shapes stand out without being trivially easy.
    let base_h = rng.f32();
    let bg_col = |rng: &mut Rng, dh: f32| {
        hsv_to_rgb(base_h + rng.uniform(-dh, dh), rng.uniform(0.05, 0.45), rng.uniform(0.2, 0.92))
    };
    let colors = [bg_col(&mut rng, 0.08), bg_col(&mut rng, 0.15), bg_col(&mut rng, 0.5)];
    let background = Background {
        kind: rng.below(4) as u8,
        colors,
        angle: rng.uniform(0.0, TAU),
        scale: rng.uniform(0.15, 0.6) * w.min(h),
        strength: rng.uniform(0.15, 0.5),
        stripe_freq: rng.uniform(0.02, 0.12),
        vignette: if rng.chance(0.5) { rng.uniform(0.1, 0.35) } else { 0.0 },
        seed: rng.next_u64(),
    };

    let n = if rng.chance(cfg.empty_fraction) || cfg.max_objects == 0 {
        0
    } else {
        cfg.min_objects + rng.below(cfg.max_objects - cfg.min_objects + 1)
    };
    let mut objects: Vec<Object> = Vec::with_capacity(n);
    let mut placed: Vec<[f32; 4]> = Vec::with_capacity(n);
    let min_side = w.min(h);
    for _ in 0..n {
        for _attempt in 0..24 {
            let k = rng.below(kinds.len());
            let r = if rng.chance(cfg.small_fraction) {
                rng.uniform(4.0, 14.0)
            } else {
                // Log-uniform between 3% and 24% of the short side.
                (min_side * 0.03) * math::exp2(3.0 * rng.f32())
            };
            let edge = rng.chance(cfg.edge_fraction);
            let m = if edge { -0.4 * r } else { r + 1.0 };
            if w - 2.0 * m <= 0.0 || h - 2.0 * m <= 0.0 {
                continue;
            }
            let c = [rng.uniform(m, w - m), rng.uniform(m, h - m)];
            let rot = rng.uniform(0.0, TAU);
            let prims = shape_prims(kinds[k], c, r, rot, &mut rng);
            let b = bounds_of(&prims);
            let clipped = [b[0].max(0.0), b[1].max(0.0), b[2].min(w), b[3].min(h)];
            if clipped[2] - clipped[0] < 3.0 || clipped[3] - clipped[1] < 3.0 {
                continue;
            }
            // The new object is drawn on top: it must not hide too much of any earlier one.
            let occludes = placed.iter().any(|o| {
                let iw = (o[2].min(clipped[2]) - o[0].max(clipped[0])).max(0.0);
                let ih = (o[3].min(clipped[3]) - o[1].max(clipped[1])).max(0.0);
                iw * ih > cfg.max_occlusion * (o[2] - o[0]) * (o[3] - o[1])
            });
            if occludes {
                continue;
            }
            let (hue, sat, val) = (rng.f32(), rng.uniform(0.45, 1.0), rng.uniform(0.35, 1.0));
            placed.push(clipped);
            objects.push(Object { class: k as u32, prims, hue, sat, val });
            break;
        }
    }
    let boxes =
        objects.iter().zip(&placed).map(|(o, b)| BBox::from_pixel_xyxy(o.class, *b, width, height)).collect();
    let grain_seed = rng.next_u64();
    Scene { width, height, background, objects, boxes, grain_seed }
}

fn render(scene: &Scene, grain: f32) -> Sample {
    let mut canvas = Canvas::new(scene.width as usize, scene.height as usize);
    scene.background.paint(&mut canvas);
    let mut rng = Rng::new(scene.grain_seed ^ 0xC010_0125);
    for o in &scene.objects {
        let b = bounds_of(&o.prims);
        let size = (b[2] - b[0]).max(b[3] - b[1]);
        let under = luma(canvas.mean_under(b));
        let mut color = hsv_to_rgb(o.hue, o.sat, o.val);
        // Keep enough contrast with what is underneath.
        if (luma(color) - under).abs() < 0.22 {
            color = if under > 0.5 {
                hsv_to_rgb(o.hue, o.sat, o.val * 0.35)
            } else {
                hsv_to_rgb(o.hue, o.sat * 0.8, (o.val + 0.45).min(1.0)).map(|v| v * 0.7 + 0.3)
            };
        }
        let off = (size * 0.035).clamp(1.0, 6.0);
        let paint = Paint {
            color,
            shade: rng.uniform(0.05, 0.2),
            shadow: ([off, off * 1.2], rng.uniform(0.15, 0.4)),
            rim: rng.uniform(0.1, 0.3),
        };
        canvas.draw(&o.prims, &paint);
    }
    Sample { image: canvas.to_image(grain, scene.grain_seed), boxes: scene.boxes.clone() }
}

/// Generates image `index` (global index over train + val) of the dataset described by `cfg`.
pub fn generate(cfg: &SyntheticConfig, index: usize) -> Result<Sample> {
    let kinds = cfg.kinds()?;
    Ok(render(&layout(cfg, &kinds, index), cfg.grain))
}

/// A synthetic split rendered on demand (labels are laid out up front, pixels on [`Dataset::get`]).
pub struct SyntheticDataset {
    cfg: SyntheticConfig,
    kinds: Vec<usize>,
    split: Split,
    start: usize,
    boxes: Vec<Vec<BBox>>,
    sizes: Vec<(u32, u32)>,
}

impl SyntheticDataset {
    /// The `split` part of the dataset `cfg` describes (the same images [`write_yolo`] writes there).
    pub fn new(cfg: SyntheticConfig, split: Split) -> Result<Self> {
        let kinds = cfg.kinds()?;
        let range = cfg.split_range(split);
        let start = range.start;
        let scenes: Vec<(Vec<BBox>, (u32, u32))> = range
            .into_par_iter()
            .map(|i| {
                let s = layout(&cfg, &kinds, i);
                (s.boxes, (s.width, s.height))
            })
            .collect();
        let (boxes, sizes) = scenes.into_iter().unzip();
        Ok(SyntheticDataset { cfg, kinds, split, start, boxes, sizes })
    }

    /// The configuration.
    pub fn config(&self) -> &SyntheticConfig {
        &self.cfg
    }
}

impl Dataset for SyntheticDataset {
    fn len(&self) -> usize {
        self.boxes.len()
    }

    fn get(&self, index: usize) -> Result<Sample> {
        if index >= self.len() {
            return Err(DataError::OutOfRange { index, len: self.len() });
        }
        Ok(render(&layout(&self.cfg, &self.kinds, self.start + index), self.cfg.grain))
    }

    fn boxes(&self, index: usize) -> &[BBox] {
        &self.boxes[index]
    }

    fn class_names(&self) -> &[String] {
        self.cfg.class_names()
    }

    fn split(&self) -> Split {
        self.split
    }

    fn image_size(&self, index: usize) -> Option<(u32, u32)> {
        self.sizes.get(index).copied()
    }
}

/// What [`write_yolo`] wrote.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WrittenDataset {
    /// Dataset root.
    pub root: PathBuf,
    /// The `data.yaml` path.
    pub yaml: PathBuf,
    /// Images per split `[train, val]`.
    pub splits: [usize; 2],
    /// Total boxes written.
    pub boxes: usize,
}

/// Writes a YOLO-format dataset to `dir`: `images/{train,val}/NNNNNN.jpg|png`, `labels/{train,val}/NNNNNN.txt` and
/// a `data.yaml` with the class names. Images are rendered and encoded in parallel; output is deterministic. `dir`
/// must be missing or empty: it is claimed before the first file ([`crate::unfinished::claim`]).
pub fn write_yolo(cfg: &SyntheticConfig, dir: &Path, format: ImageFormat) -> Result<WrittenDataset> {
    cfg.check()?;
    write_yolo_with_progress(cfg, crate::unfinished::claim(dir)?, format, |_| {})
}

/// [`write_yolo`] into a folder the caller claimed, that calls `progress(done)` after each image is written. Calls
/// come from worker threads and `done` counts up to `cfg.images`. Check `cfg` first ([`SyntheticConfig::check`]):
/// like any error here, a refused config leaves the folder marked unfinished.
pub fn write_yolo_with_progress(
    cfg: &SyntheticConfig,
    out: crate::unfinished::Claim,
    format: ImageFormat,
    progress: impl Fn(usize) + Sync,
) -> Result<WrittenDataset> {
    let done = std::sync::atomic::AtomicUsize::new(0);
    let kinds = cfg.kinds()?;
    let dir = out.dir();
    for split in ["train", "val"] {
        for kind in ["images", "labels"] {
            let d = dir.join(kind).join(split);
            std::fs::create_dir_all(&d).map_err(|e| DataError::io(&d, e))?;
        }
    }
    let ntrain = cfg.train_count();
    let boxes: Vec<usize> = (0..cfg.images)
        .into_par_iter()
        .map(|i| -> Result<usize> {
            let split = if i < ntrain { "train" } else { "val" };
            let s = render(&layout(cfg, &kinds, i), cfg.grain);
            let name = format!("{i:06}");
            let img_path = dir.join("images").join(split).join(format!("{name}.{}", format.extension()));
            write_image(&s.image, &img_path, format)?;
            let mut text = String::with_capacity(s.boxes.len() * 40);
            for b in &s.boxes {
                use std::fmt::Write;
                let _ = writeln!(text, "{} {:.6} {:.6} {:.6} {:.6}", b.class, b.cx, b.cy, b.w, b.h);
            }
            let lbl = dir.join("labels").join(split).join(format!("{name}.txt"));
            std::fs::write(&lbl, text).map_err(|e| DataError::io(&lbl, e))?;
            progress(done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1);
            Ok(s.boxes.len())
        })
        .collect::<Result<_>>()?;
    let mut yaml = format!(
        "# FOLO synthetic shapes dataset: {} images, seed {}, imgsz {}\ntrain: images/train\nval: images/val\nnames:\n",
        cfg.images, cfg.seed, cfg.imgsz
    );
    for (i, c) in cfg.class_names().iter().enumerate() {
        // JSON strings are valid YAML double-quoted scalars, so any name reads back unchanged.
        yaml.push_str(&format!("  {i}: {}\n", serde_json::to_string(c).expect("strings serialize")));
    }
    let yaml_path = dir.join("data.yaml");
    std::fs::write(&yaml_path, yaml).map_err(|e| DataError::io(&yaml_path, e))?;
    let root = dir.to_path_buf();
    out.finish()?;
    Ok(WrittenDataset {
        root,
        yaml: yaml_path,
        splits: [ntrain, cfg.images - ntrain],
        boxes: boxes.iter().sum(),
    })
}

#[cfg(test)]
mod tests;
