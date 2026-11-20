//! Golden batches: a digest of every batch the loader makes for a fixed seed, for the training transform in each of
//! its paths (mosaic, mosaic with rotation and shear, mosaic with mixup and up-down flips, letterbox + affine) and
//! for validation, from images in memory and from JPEG and PNG files. They pin the loader's output bit for bit, so
//! an optimisation of the per-sample path (decode, resize, mosaic, warp, HSV, flips, collate) proves itself
//! identical on every platform CI runs (Linux and Windows) and every worker count (docs/DATA.md, DataLoader and
//! determinism).
//!
//! A failure here means the loader makes other batches than it did. If that is intended (a change of the
//! augmentation's numerics, which needs its own accuracy check: docs/notes/hill/r6-loader.md), re-bless the values
//! below with the ones the failure prints, and say so in the pull request.

mod common;

use common::TempDir;
use folo_data::{
    AugmentConfig, Batch, DataLoader, Dataset, Image, ImageFormat, InMemoryDataset, LoaderConfig, Rng,
    Sample, Split, encode, open,
};
use std::sync::Arc;

const RE_BLESS: &str = "the loader's batches changed: if this is intended, re-bless tests/golden.rs with the \
                        values printed here and say so in the pull request";

/// FNV-1a over a stream of bytes: a stable 64-bit digest.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xCBF2_9CE4_8422_2325)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01B3);
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// The digest of one batch: pixels (`u8` or the `f32` bits), shape, targets and meta.
fn digest(b: &Batch) -> u64 {
    let mut h = Fnv::new();
    h.bytes(&b.images_u8);
    h.f32s(&b.images);
    for &d in &b.shape {
        h.u64(d as u64);
    }
    for t in &b.targets {
        h.f32s(t);
    }
    for m in &b.meta {
        h.u64(m.index as u64);
        h.u64(m.orig_size[0] as u64);
        h.u64(m.orig_size[1] as u64);
        h.f32s(&m.ratio);
        h.f32s(&m.pad);
    }
    h.0
}

/// A textured `w x h` image: smooth gradients, a hashed grain, a checkerboard and saturated patches, so every
/// interpolation weight and every HSV sector shows in the bytes.
fn texture(seed: u64, w: u32, h: u32) -> Image {
    let mut img = Image::new(w, h, [0, 0, 0]);
    for y in 0..h {
        for x in 0..w {
            let grain = folo_data::splitmix64(&mut (seed ^ ((y as u64) << 32 | x as u64)));
            let check = ((x / 7 + y / 5) % 2) * 50;
            let p = [
                ((x * 255 / w.max(1)) + check + (grain & 31) as u32) as u8,
                ((y * 200 / h.max(1)) + (grain >> 8 & 63) as u32) as u8,
                (((x + y) * 3 + check) as u8).wrapping_add((grain >> 16) as u8 & 15),
            ];
            img.set_pixel(x, y, p);
        }
    }
    let mut rng = Rng::new(seed);
    for _ in 0..3 {
        let (x0, y0) = (rng.below(w as usize) as i64, rng.below(h as usize) as i64);
        let c = [[240, 20, 30], [10, 220, 40], [30, 40, 230], [250, 250, 0], [128, 128, 128]][rng.below(5)];
        img.fill_rect(
            x0,
            y0,
            x0 + 1 + rng.below(w as usize / 2) as i64,
            y0 + 1 + rng.below(h as usize / 2) as i64,
            c,
        );
    }
    img
}

/// Sizes larger and smaller than the network size, both orientations, odd sides.
const SIZES: [(u32, u32); 7] = [(200, 120), (50, 90), (96, 96), (130, 31), (33, 200), (97, 61), (321, 240)];

fn sample(k: usize) -> Sample {
    let (w, h) = SIZES[k % SIZES.len()];
    let mut s = common::random_rect_sample(k as u64, w, h, 3);
    s.image = texture(k as u64, w, h);
    s
}

fn memory_dataset(n: usize) -> Arc<dyn Dataset> {
    Arc::new(InMemoryDataset {
        samples: (0..n).map(sample).collect(),
        classes: (0..4).map(|c| c.to_string()).collect(),
        split: Split::Train,
    })
}

/// The same images written as files (JPEG quality 90, every third one PNG), with their boxes as YOLO labels.
fn file_dataset(dir: &TempDir, n: usize) -> Arc<dyn Dataset> {
    for k in 0..n {
        let s = sample(k);
        let (ext, fmt) = if k % 3 == 2 { ("png", ImageFormat::Png) } else { ("jpg", ImageFormat::Jpeg(90)) };
        common::write(&dir.join(format!("images/train/{k:02}.{ext}")), encode(&s.image, fmt).unwrap());
        let labels: String =
            s.boxes.iter().map(|b| format!("{} {} {} {} {}\n", b.class, b.cx, b.cy, b.w, b.h)).collect();
        common::write(&dir.join(format!("labels/train/{k:02}.txt")), labels);
    }
    common::write(
        &dir.join("data.yaml"),
        "path: .\ntrain: images/train\nval: images/train\nnames: [a, b, c, d]\n",
    );
    Arc::new(open(dir.path()).unwrap().split(Split::Train).clone())
}

fn train(augment: AugmentConfig) -> LoaderConfig {
    LoaderConfig {
        batch_size: 4,
        imgsz: 96,
        workers: 2,
        seed: 11,
        augment: Some(augment),
        u8_images: true,
        ..LoaderConfig::default()
    }
}

/// The configurations pinned, by name.
fn configs() -> Vec<(&'static str, LoaderConfig)> {
    let default = AugmentConfig::default();
    vec![
        ("mosaic", train(default)),
        ("mosaic f32", LoaderConfig { u8_images: false, ..train(default) }),
        ("mosaic rotate shear", train(AugmentConfig { degrees: 10.0, shear: 3.0, ..default })),
        ("mosaic mixup flipud", train(AugmentConfig { mixup: 0.5, flipud: 0.5, ..default })),
        ("letterbox affine", train(AugmentConfig { mosaic: 0.0, ..default })),
        ("letterbox rotate", train(AugmentConfig { mosaic: 0.0, degrees: 7.0, ..default })),
        ("validation", LoaderConfig { augment: None, shuffle: false, ..train(default) }),
    ]
}

/// Digests of the first three batches of epoch 1 for each configuration, from the dataset in memory.
const MEMORY: [[u64; 3]; 7] = [
    [0x5ce2a31e57129d69, 0xe104be4e3207f810, 0x07cf6fcb693f9bed], // mosaic
    [0x6945f9cf19b6fe78, 0xf206d88d1ad958ea, 0x6ed911a0b1f2d839], // mosaic f32
    [0x13c3cf193f1f0b08, 0xec65bce9aec67241, 0x50a12c7c7e9844ea], // mosaic rotate shear
    [0xd7377d82bd5a84c7, 0xa42a61fcd51832ba, 0xc8f3dc120459fc20], // mosaic mixup flipud
    [0x88761083e2583a4e, 0x0af8c06fb8af7a13, 0xa41849ee9dfd1ff4], // letterbox affine
    [0x2632101e15d4ced5, 0x2b575568868497ae, 0x4231f41091d0a539], // letterbox rotate
    [0xea623c3047d63d32, 0x37bb43098ebcf986, 0xfcd39d551b839d21], // validation
];

/// The same from the image files.
const FILES: [[u64; 3]; 7] = [
    [0x769143e2f27ab3cb, 0x97d51be981ea7f47, 0xdc4387d1c26d1c43], // mosaic
    [0xd9fdb5048d2a9ff6, 0x68eb6ed52df6c114, 0xf1e9c01eb427c98f], // mosaic f32
    [0xfd2a20e1bb1f6727, 0x8bbf505daeb6dd50, 0x94a09245c722c329], // mosaic rotate shear
    [0xcd9645145c5f1ba7, 0xca6d7316fb4d23f8, 0x55559713b8e0047d], // mosaic mixup flipud
    [0x4b090b7071e843d6, 0x2ad1b5c16a255478, 0x39323e91ed62f131], // letterbox affine
    [0xfef88d020ae5523b, 0xd1bf16cd822f335b, 0xfc03b06dee416342], // letterbox rotate
    [0x16a2a94ab03abb96, 0x883c7ab7b77b851f, 0x3676ef8b3ddcaaa7], // validation
];

/// Digests of the first three batches of epoch 1 for each of `configs`.
fn digests(ds: &Arc<dyn Dataset>, configs: &[(&'static str, LoaderConfig)]) -> Vec<(&'static str, [u64; 3])> {
    configs
        .iter()
        .map(|(name, cfg)| {
            let loader = DataLoader::new(ds.clone(), cfg.clone()).unwrap();
            let d: Vec<u64> = loader.epoch(1).take(3).map(|b| digest(&b.unwrap())).collect();
            (*name, [d[0], d[1], d[2]])
        })
        .collect()
}

fn assert_golden(got: &[(&str, [u64; 3])], golden: &[[u64; 3]], what: &str) {
    let found: Vec<[u64; 3]> = got.iter().map(|(_, d)| *d).collect();
    if found != golden {
        let mut msg = format!(
            "{RE_BLESS}
{what}: [
"
        );
        for (name, d) in got {
            msg += &format!(
                "    [{:#018x}, {:#018x}, {:#018x}], // {name}
",
                d[0], d[1], d[2]
            );
        }
        panic!("{msg}]");
    }
}

fn check(ds: Arc<dyn Dataset>, golden: &[[u64; 3]; 7], what: &str) {
    assert_golden(&digests(&ds, &configs()), golden, what);
}

#[test]
fn batches_from_memory_are_golden() {
    check(memory_dataset(14), &MEMORY, "MEMORY");
}

#[test]
fn batches_from_files_are_golden() {
    let dir = TempDir::new("golden");
    check(file_dataset(&dir, 14), &FILES, "FILES");
}

/// At imgsz 192 from the files: the letterboxes, the warped samples, most resized tiles and the larger decoded images
/// are large enough for the loader's buffer pool (64 KiB and up; the 50 x 90 image and its resized tile are not),
/// where imgsz 96 leaves all but the mosaic canvas to plain allocations.
const LARGE: [[u64; 3]; 4] = [
    [0xa4efaa43906f678b, 0xf00810c7dd3675e8, 0x80eacf6d3e6c7bd0], // mosaic
    [0x26f1f3d83c43b64d, 0x24b33b064b006555, 0x8779b930596dfee4], // mosaic rotate shear
    [0xf4f4561b2004e1f1, 0x030bcfc33f154304, 0x9b82352ea4240c26], // letterbox affine
    [0x3602c5be7c80e213, 0x5334afa76a869a38, 0x4a39ce535f8dd160], // validation
];

/// The pooled sizes give the same batches on the calling thread (no workers, no pool) and on 3 workers (pooled
/// buffers, reused across samples and threads), and those are golden.
#[test]
fn large_batches_are_golden_with_and_without_workers() {
    let dir = TempDir::new("golden-large");
    let ds = file_dataset(&dir, 14);
    let pinned = ["mosaic", "mosaic rotate shear", "letterbox affine", "validation"];
    for workers in [0, 3] {
        let configs: Vec<_> = configs()
            .into_iter()
            .filter(|(name, _)| pinned.contains(name))
            .map(|(name, cfg)| (name, LoaderConfig { imgsz: 192, workers, ..cfg }))
            .collect();
        assert_golden(&digests(&ds, &configs), &LARGE, &format!("LARGE ({workers} workers)"));
    }
}
