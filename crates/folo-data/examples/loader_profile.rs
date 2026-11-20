//! Splits a training sample's time and allocations into the stages of the training transform, single-threaded, on
//! a real dataset (docs/notes/hill/r6-loader.md).
//!
//! ```sh
//! SCAN_CACHE=D:/scan-cache cargo run -p folo-data --release --example loader_profile -- D:/folo-data/coco [N] [IMGSZ]
//! ```
//!
//! Runs `N` (default 256) mosaic samples of the default training augmentation at `IMGSZ` (default 640), each drawing
//! its images and parameters as the training transform does (`TrainTransform::apply` with mosaic on), after one
//! discarded pass over the same files (so every stage reads from a warm page cache). Prints, per stage, the time per
//! sample, its share, and the allocations and bytes allocated per sample.

use folo_data::augment::{flip_lr, flip_ud, mosaic4, random_affine, random_hsv, resize_long_side};
use folo_data::{AugmentConfig, Dataset, Rng, Sample, Split, decode};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::process::ExitCode;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

struct Counting;

static ALLOCS: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);

// SAFETY: forwards to the system allocator; only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        // SAFETY: the caller's contract is the system allocator's.
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(layout.size() as u64, Relaxed);
        // SAFETY: as above.
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: as above.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Relaxed);
        BYTES.fetch_add(new_size as u64, Relaxed);
        // SAFETY: as above.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

const STAGES: [&str; 9] = [
    "file read",
    "JPEG decode",
    "resize long side",
    "mosaic canvas",
    "affine warp",
    "HSV",
    "flips",
    "collate u8",
    "total",
];

#[derive(Default, Clone, Copy)]
struct Acc {
    secs: f64,
    allocs: u64,
    bytes: u64,
}

struct Timer<'a> {
    acc: &'a mut [Acc; 9],
}

impl Timer<'_> {
    fn stage<T>(&mut self, k: usize, f: impl FnOnce() -> T) -> T {
        let (a0, b0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed));
        let t = Instant::now();
        let out = f();
        let s = t.elapsed().as_secs_f64();
        let e = &mut self.acc[k];
        e.secs += s;
        e.allocs += ALLOCS.load(Relaxed) - a0;
        e.bytes += BYTES.load(Relaxed) - b0;
        out
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: loader_profile PATH [N] [IMGSZ]   (env SCAN_CACHE=DIR turns on the scan cache)");
        return ExitCode::from(2);
    };
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(256);
    let s: u32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(640);
    if let Some(dir) = std::env::var_os("SCAN_CACHE").filter(|d| !d.is_empty()) {
        folo_data::set_scan_cache_dir(Some(dir.into()));
    }
    let t = Instant::now();
    let ds = match folo_data::open(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("loader_profile: {e}");
            return ExitCode::FAILURE;
        }
    };
    let train = ds.split(Split::Train).clone();
    let len = train.len();
    println!(
        "{path}: {len} train images, opened in {:.1} s; {n} mosaic samples at {s}\n",
        t.elapsed().as_secs_f64()
    );
    let cfg = AugmentConfig::default();
    // The images each sample uses, drawn as `TrainTransform` draws them (mosaic chance, then three partners).
    let plan: Vec<(Rng, [usize; 4])> = (0..n)
        .map(|p| {
            let mut rng = Rng::derive(&[0, 0, p as u64]);
            let _ = rng.chance(cfg.mosaic);
            let ids = [p * 7919 % len, rng.below(len), rng.below(len), rng.below(len)];
            (rng, ids)
        })
        .collect();
    // Warm the page cache with the same files.
    for (_, ids) in &plan {
        for &i in ids {
            let _ = black_box(std::fs::read(train.image_path(i).unwrap()));
        }
    }
    let mut acc = [Acc::default(); 9];
    let mut pixels = 0u64;
    folo_data::par::serial(|| {
        for (rng, ids) in &plan {
            let mut rng = rng.clone();
            let mut tm = Timer { acc: &mut acc };
            let (a0, b0, t0) = (ALLOCS.load(Relaxed), BYTES.load(Relaxed), Instant::now());
            let mut parts = Vec::with_capacity(4);
            for &i in ids {
                let bytes = tm.stage(0, || std::fs::read(train.image_path(i).unwrap()).unwrap());
                let image = tm.stage(1, || decode(&bytes).unwrap());
                pixels += image.width as u64 * image.height as u64;
                let sample = Sample { image, boxes: train.boxes(i).to_vec() };
                parts.push(tm.stage(2, || resize_long_side(sample, s)));
            }
            let canvas = tm.stage(3, || mosaic4([&parts[0], &parts[1], &parts[2], &parts[3]], s, &mut rng));
            let mut out = tm.stage(4, || random_affine(&canvas, s, s, &cfg, &mut rng));
            let _ = rng.chance(cfg.mixup);
            tm.stage(5, || random_hsv(&mut out.image, cfg.hsv, &mut rng));
            tm.stage(6, || {
                if rng.chance(cfg.flipud) {
                    flip_ud(&mut out);
                }
                if rng.chance(cfg.fliplr) {
                    flip_lr(&mut out);
                }
            });
            tm.stage(7, || {
                let plane = (s * s) as usize;
                let mut o = vec![0u8; 3 * plane];
                let (r, rest) = o.split_at_mut(plane);
                let (g, b) = rest.split_at_mut(plane);
                for (i, px) in out.image.data.as_chunks::<3>().0.iter().enumerate() {
                    r[i] = px[0];
                    g[i] = px[1];
                    b[i] = px[2];
                }
                black_box(o);
            });
            let e = &mut acc[8];
            e.secs += t0.elapsed().as_secs_f64();
            e.allocs += ALLOCS.load(Relaxed) - a0;
            e.bytes += BYTES.load(Relaxed) - b0;
        }
    });
    let total = acc[8].secs;
    println!("mean decoded image: {:.0} kpx\n", pixels as f64 / (4 * n) as f64 / 1e3);
    println!("| stage | ms / sample | share | allocs / sample | MB allocated / sample |");
    println!("|---|---|---|---|---|");
    for (k, name) in STAGES.iter().enumerate() {
        let a = acc[k];
        println!(
            "| {name} | {:.3} | {:.1}% | {:.1} | {:.2} |",
            a.secs * 1e3 / n as f64,
            100.0 * a.secs / total,
            a.allocs as f64 / n as f64,
            a.bytes as f64 / n as f64 / 1e6
        );
    }
    ExitCode::SUCCESS
}
